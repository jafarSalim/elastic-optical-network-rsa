"""Deep reinforcement learning RMSA agent (PPO with action masking).

The agent sees, for each of the k candidate paths, 13 numbers (reach and
feasibility flags, hops, length, format, slots needed, free spectrum on the
path, the number and size of blocks that fit, and two placements: first fit
and the in-path placement that cuts fewest fragments, each with its start slot
and cut count). It picks one of 2k actions = (path, placement). Infeasible
actions are masked; when nothing fits the request is blocked without a
decision.

Reward: +1 for every request it places, minus 1 for every request that is
blocked before its next decision. The return therefore counts accepted
requests, so the agent learns to keep spectrum usable for future demands.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .policies import placements
from .sim import Simulator

F_PATH = 13


class RMSAEnv:
    def __init__(self, net, load, S=320, seed=0):
        self.net, self.load, self.S, self.seed = net, load, S, seed
        self.k = net.k
        self.n_actions = 2 * self.k
        self.obs_dim = F_PATH * self.k + 4

    def reset(self, seed=None):
        self.sim = Simulator(self.net, self.load, S=self.S,
                             seed=self.seed if seed is None else seed)
        self.blocked_auto = 0
        return self._advance()

    def _advance(self):
        """Move to the next request that has at least one feasible action."""
        penalty = 0
        while True:
            s, d, rate, hold = self.sim.next_request()
            cands = self.sim.candidates(s, d, rate)
            if any(c.starts.size for c in cands):
                self.cur = (s, d, rate, hold, cands)
                obs, mask, self.acts = self._features(rate, cands)
                self.pending_penalty = penalty
                return obs, mask
            self.sim.record(rate, True, self.net.paths[(s, d)][0].hops)
            penalty += 1

    def _features(self, rate, cands):
        sp = self.sim.spec
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        mask = np.zeros(self.n_actions, dtype=bool)
        acts = [None] * self.n_actions
        for c in cands:
            k = c.path_idx
            ids = c.path.link_ids
            o = obs[k * F_PATH:(k + 1) * F_PATH]
            o[0] = 1.0
            o[2] = c.path.hops / 10
            o[3] = c.path.km / 8000
            o[4] = c.fmt["bits"] / 6
            o[5] = c.need / 25
            o[6] = sp.path_free(ids).mean()
            if c.starts.size == 0:
                continue
            o[1] = 1.0
            o[7] = min(c.starts.size, 20) / 20
            o[8] = c.lens.max() / self.S
            ff = int(c.starts[0])
            o[9] = ff / self.S
            o[10] = sp.cuts(ids, ff, c.need) / (2 * c.path.hops)
            pl = placements(c)
            cuts = np.array([sp.cuts(ids, int(p), c.need) for p in pl])
            j = int(np.argmin(cuts))
            o[11] = pl[j] / self.S
            o[12] = cuts[j] / (2 * c.path.hops)
            mask[2 * k] = mask[2 * k + 1] = True
            acts[2 * k] = (c, ff)
            acts[2 * k + 1] = (c, int(pl[j]))
        base = F_PATH * self.k
        obs[base + RATE_IDX[rate]] = 1.0
        obs[base + 3] = sp.utilisation()
        return obs, mask, acts

    def step(self, a):
        s, d, rate, hold, _ = self.cur
        cand, start = self.acts[a]
        self.sim.commit(cand, start, rate, hold)
        self.sim.record(rate, False, self.net.paths[(s, d)][0].hops)
        obs, mask = self._advance()
        return obs, mask, 1.0 - self.pending_penalty


RATE_IDX = {100: 0, 200: 1, 400: 2}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        h = self.body(x)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


def train_ppo(net, load, total_steps=300_000, rollout=4096, epochs=4, minibatch=512,
              gamma=0.95, lam=0.95, lr=3e-4, clip=0.2, ent=0.01, seed=0, log=print,
              episode_len=20_000):
    torch.manual_seed(seed)
    env = RMSAEnv(net, load, seed=seed)
    model = ActorCritic(env.obs_dim, env.n_actions)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs, mask = env.reset(seed=seed * 1000)
    ep_steps, ep = 0, 0
    curve = []
    t0 = time.time()
    steps = 0
    while steps < total_steps:
        O = np.zeros((rollout, env.obs_dim), np.float32)
        M = np.zeros((rollout, env.n_actions), bool)
        A = np.zeros(rollout, np.int64)
        R = np.zeros(rollout, np.float32)
        V = np.zeros(rollout + 1, np.float32)
        LP = np.zeros(rollout, np.float32)
        for t in range(rollout):
            with torch.no_grad():
                lg, v = model(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
                dist = torch.distributions.Categorical(logits=lg)
                a = dist.sample()
            O[t], M[t], A[t], V[t], LP[t] = obs, mask, a.item(), v.item(), dist.log_prob(a).item()
            obs, mask, r = env.step(int(a))
            R[t] = r
            ep_steps += 1
            if ep_steps >= episode_len:      # restart traffic from an empty network
                st = env.sim.stats
                curve.append((steps + t + 1, st["blocked"] / max(st["arrived"], 1)))
                log(f"  step {steps + t + 1:>7}: episode blocking {curve[-1][1]:.4f}"
                    f"  ({time.time() - t0:.0f} s)")
                ep += 1
                obs, mask = env.reset(seed=seed * 1000 + ep)
                ep_steps = 0
        with torch.no_grad():
            V[rollout] = model(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])[1].item()
        adv = np.zeros(rollout, np.float32)
        g = 0.0
        for t in reversed(range(rollout)):
            delta = R[t] + gamma * V[t + 1] - V[t]
            g = delta + gamma * lam * g
            adv[t] = g
        ret = adv + V[:-1]
        tO, tM, tA = map(torch.from_numpy, (O, M, A))
        tLP, tAdv, tRet = map(torch.from_numpy, (LP, adv, ret))
        tAdv = (tAdv - tAdv.mean()) / (tAdv.std() + 1e-8)
        for _ in range(epochs):
            perm = torch.randperm(rollout)
            for i in range(0, rollout, minibatch):
                b = perm[i:i + minibatch]
                lg, v = model(tO[b], tM[b])
                dist = torch.distributions.Categorical(logits=lg)
                ratio = torch.exp(dist.log_prob(tA[b]) - tLP[b])
                s1 = ratio * tAdv[b]
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * tAdv[b]
                loss = -torch.min(s1, s2).mean() + 0.5 * ((v - tRet[b]) ** 2).mean() \
                    - ent * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
        steps += rollout
    return model, curve


def evaluate(net, model, load, n_requests=30000, warmup=5000, seed=0):
    """Greedy agent on the same traffic as sim.run(seed=seed)."""
    env = RMSAEnv(net, load, seed=seed)
    sim = Simulator(net, load, seed=seed)
    env.sim = sim
    model.eval()
    util = []
    for i in range(warmup + n_requests):
        if i == warmup:
            sim.reset_stats()
        s, d, rate, hold = sim.next_request()
        cands = sim.candidates(s, d, rate)
        hops = net.paths[(s, d)][0].hops
        if not any(c.starts.size for c in cands):
            sim.record(rate, True, hops)
            continue
        obs, mask, acts = env._features(rate, cands)
        with torch.no_grad():
            lg, _ = model(torch.from_numpy(obs)[None], torch.from_numpy(mask)[None])
        cand, start = acts[int(lg.argmax())]
        sim.commit(cand, start, rate, hold)
        sim.record(rate, False, hops)
        if i >= warmup and i % 100 == 0:
            util.append(sim.spec.utilisation())
    st = sim.stats
    return dict(bp=st["blocked"] / st["arrived"], bbp=st["bw_blocked"] / st["bw_arrived"],
                blocked=st["blocked"], arrived=st["arrived"], util=float(np.mean(util)))
