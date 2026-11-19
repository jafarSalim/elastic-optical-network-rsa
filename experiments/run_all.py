"""Run every experiment, save results/*.json|csv and figures/*.png.

    python experiments/run_all.py                 # all (about 45 min on 2 cores)
    python experiments/run_all.py physics maps    # only some

The DRL experiment needs the models from experiments/train_drl.py.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import style  # noqa: E402
from eon import physics  # noqa: E402
from eon.policies import POLICIES, make_fragmentation_aware  # noqa: E402
from eon.sim import run  # noqa: E402
from eon.topology import load  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
SEEDS = [11, 12, 13, 14, 15]
N_REQ, WARM = 20000, 5000
LOADS = {"nsfnet": [100, 120, 140, 160, 180, 200, 220, 240],
         "germany50": [200, 240, 280, 320, 360, 400, 440, 480]}
NETS = {}


def net_of(name, k=5):
    if (name, k) not in NETS:
        NETS[(name, k)] = load(name, k=k)
    return NETS[(name, k)]


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def save_json(name, obj):
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)
    print("  figure", name)


# 1 -------------------------------------------------------------------------
def exp_physics():
    p = physics.LinkParams()
    table = physics.reach_table(p)
    launch = np.linspace(-12, 6, 181)
    spans = np.arange(1, 121)
    out = dict(optimum_launch_dbm=physics.optimum_launch_dbm(p), table=table,
               gsnr_vs_launch={n: physics.gsnr_db(n, p, launch).tolist() for n in (1, 10, 40)},
               launch=launch.tolist(), spans=spans.tolist(), gsnr=physics.gsnr_db(spans, p).tolist())
    save_json("exp1_physics", out)
    with open(os.path.join(RES, "reach_table.csv"), "w") as f:
        f.write("format,bits_per_symbol,snr_required_db,with_margin_db,max_spans,reach_km,"
                "slots_100G,slots_200G,slots_400G\n")
        for r in table:
            f.write(f"{r['name']},{r['bits']},{r['snr_req_db']:.2f},{r['need_db']:.2f},"
                    f"{r['max_spans']},{r['reach_km']:.0f},"
                    + ",".join(str(physics.slots_needed(x, r['bits'])) for x in (100, 200, 400)) + "\n")

    style.apply()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cols = ["#22d3ee", "#facc15", "#ff4d9d"]
    for c, n in zip(cols, (1, 10, 40)):
        g = np.array(out["gsnr_vs_launch"][n])
        ax[0].plot(launch, g, color=c, label=f"{n} span{'s' if n > 1 else ''} ({n * 80} km)")
        i = int(np.argmax(g))
        ax[0].plot(launch[i], g[i], "o", color=c, ms=8, mec="white")
    ax[0].axvline(out["optimum_launch_dbm"], color=style.MUTED, ls=":", lw=1.2)
    ax[0].text(out["optimum_launch_dbm"] + 0.3, 24.2,
               f"optimum {out['optimum_launch_dbm']:.1f} dBm per 32 GBd channel", color=style.MUTED, fontsize=9)
    ax[0].set_ylim(top=25.5)
    ax[0].set(xlabel="launch power per channel (dBm)", ylabel="GSNR (dB)",
              title="ASE rises to the left, Kerr noise to the right")
    ax[0].legend(loc="lower center")
    km = spans * 80
    ax[1].plot(km, out["gsnr"], color="white", lw=2.6, label="GSNR at optimum launch")
    for r in table:
        c = style.FORMAT_COLORS[r["name"]]
        ax[1].axhline(r["need_db"], color=c, lw=1.3, ls="--")
        lab = f"{r['name']}  {r['need_db']:.1f} dB  →  " + (f"{r['reach_km']:.0f} km" if r["reach_km"] else "never")
        ax[1].text(9500, r["need_db"] + 0.25, lab, color=c, fontsize=9, ha="right")
        if r["reach_km"]:
            ax[1].plot(r["reach_km"], r["need_db"], "o", color=c, ms=7, mec="white")
    ax[1].set(xlabel="path length (km, 80 km spans)", ylabel="dB", xlim=(0, 9600), ylim=(5, 27),
              title="Reach of each format (BER 3.8e-3 + 2 dB margin)")
    ax[1].legend(loc="lower left")
    style.title(fig, "Physical layer: closed-form GN model sets the modulation reach")
    fig.tight_layout()
    savefig(fig, "fig1_physics")


# 2 -------------------------------------------------------------------------
def exp_maps():
    style.apply()
    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.25, wspace=0.12)
    summary = {}
    names = [r["name"] for r in physics.reach_table()]
    for row, name in enumerate(["nsfnet", "germany50"]):
        net = net_of(name)
        ax = fig.add_subplot(gs[row, 0])
        km = net.link_km
        cm = plt.get_cmap("plasma")
        for (u, v), L in zip(net.links, km):
            (x1, y1), (x2, y2) = net.pos[u], net.pos[v]
            c = cm((L - km.min()) / (km.max() - km.min()))
            ax.plot([x1, x2], [y1, y2], color=c, lw=2.6 if name == "nsfnet" else 1.8, zorder=1)
            if name == "nsfnet":
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{L:.0f}", fontsize=7.5, color=style.FG,
                        ha="center", va="center", bbox=dict(fc=style.PANEL, ec="none", pad=0.6))
        xs = np.array([net.pos[n] for n in net.graph.nodes])
        deg = np.array([net.graph.degree(n) for n in net.graph.nodes])
        ax.scatter(xs[:, 0], xs[:, 1], s=40 + 30 * deg, c="#22d3ee", edgecolors="white", zorder=3)
        if name == "nsfnet":
            off = {9: (0, -1.5), 13: (1.2, 0.9), 14: (0.8, -1.4), 12: (0, 0.9), 8: (0, -1.5),
                   7: (0, 0.9), 10: (1.2, -1.2), 6: (0, -1.6)}
            for n in net.graph.nodes:
                x, y = net.pos[n]
                dx, dy = off.get(n, (0, 0.9))
                ax.text(x + dx, y + dy, net.labels[n], fontsize=8.5, ha="center", color=style.FG,
                        fontweight="bold")
            ax.set_xlim(-125, -71)
        else:
            ax.set_aspect(1.5)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{net.name}: {net.n_nodes} nodes, {net.n_links} links, "
                     f"links {km.min():.0f}–{km.max():.0f} km (colour)")
        # format mix of shortest paths
        ax2 = fig.add_subplot(gs[row, 1])
        picker = physics.ModulationPicker()
        L0 = np.array([p[0].km for p in net.paths.values()])
        fm = [picker.pick(p[0].spans) for p in net.paths.values()]
        fm = np.array([f["name"] if f else "none" for f in fm])
        bins = np.linspace(0, L0.max() * 1.02, 22)
        bottom = np.zeros(len(bins) - 1)
        mix = {}
        for f in names:
            h, _ = np.histogram(L0[fm == f], bins)
            mix[f] = int((fm == f).sum())
            if h.sum():
                ax2.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom, align="edge",
                        color=style.FORMAT_COLORS[f], label=f"{f} ({100 * (fm == f).mean():.0f} %)",
                        edgecolor=style.BG, lw=0.4)
                bottom += h
        ax2.set(xlabel="shortest-path length (km)", ylabel="node pairs",
                title="Format of each shortest path")
        ax2.legend(fontsize=9)
        npaths = np.array([len([q for q in p if picker.pick(q.spans)]) for p in net.paths.values()])
        summary[name] = dict(nodes=net.n_nodes, links=net.n_links, link_km=[float(km.min()), float(km.max())],
                             shortest_path_km=dict(min=float(L0.min()), median=float(np.median(L0)),
                                                   max=float(L0.max())),
                             format_mix=mix, mean_feasible_paths=float(npaths.mean()))
    style.title(fig, "Two reference networks",
                "NSFNET with the DeepRMSA link lengths; Germany50 (SNDlib) with great-circle distance × 1.25")
    savefig(fig, "fig2_networks")
    save_json("exp2_networks", summary)


# 3 -------------------------------------------------------------------------
def _job(args):
    name, pol, L, seed = args
    r = run(net_of(name), POLICIES[pol], L, n_requests=N_REQ, warmup=WARM, seed=seed)
    r.pop("trace")
    return (name, pol, L, seed, r)


def exp_blocking(procs=1):
    jobs = [(n, p, L, s) for n in LOADS for L in LOADS[n] for p in POLICIES for s in SEEDS]
    t0 = time.time()
    with Pool(procs) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    print(f"  {len(jobs)} runs in {time.time() - t0:.0f} s")
    out = {}
    for name, pol, L, seed, r in res:
        out.setdefault(name, {}).setdefault(pol, {}).setdefault(str(L), []).append(r)
    save_json("exp3_blocking_raw", out)
    rows = ["network,policy,load_erlang,bp_mean,bp_ci95,bbp_mean,bbp_ci95,util_mean,frag_mean,blocked_total"]
    for name in out:
        for pol in out[name]:
            for L, rs in out[name][pol].items():
                bp = [r["bp"] for r in rs]
                bbp = [r["bbp"] for r in rs]
                rows.append(f"{name},{pol},{L},{np.mean(bp):.6f},{ci95(bp):.6f},{np.mean(bbp):.6f},"
                            f"{ci95(bbp):.6f},{np.mean([r['util'] for r in rs]):.4f},"
                            f"{np.mean([r['frag'] for r in rs]):.4f},{sum(r['blocked'] for r in rs)}")
    with open(os.path.join(RES, "exp3_blocking.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    fig_blocking()


def fig_blocking():
    out = load_json("exp3_blocking_raw")
    style.apply()
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for a, name in zip(ax, ["nsfnet", "germany50"]):
        for pol in POLICIES:
            Ls = LOADS[name]
            m = np.array([np.mean([r["bp"] for r in out[name][pol][str(L)]]) for L in Ls])
            e = np.array([ci95([r["bp"] for r in out[name][pol][str(L)]]) for L in Ls])
            ok = m > 0
            a.errorbar(np.array(Ls)[ok], m[ok], yerr=e[ok], color=style.COLORS[pol],
                       marker=style.MARKERS[pol], ms=7 if pol != "Frag-Aware" else 11,
                       capsize=3, label=pol, mec="white", mew=0.6)
        a.set_yscale("log")
        a.set(xlabel="offered load (Erlang)", ylabel="request blocking probability",
              title=f"{net_of(name).name}", ylim=(1e-4, 0.4))
        a.legend(loc="lower right")
    style.title(fig, "Blocking probability vs load",
                f"mean of {len(SEEDS)} seeds × {N_REQ:,} requests (after {WARM:,} warm-up), bars = 95 % CI; "
                "100/200/400 Gb/s, 320 slots of 12.5 GHz")
    fig.tight_layout()
    savefig(fig, "fig3_blocking")

    # fairness at one load per network
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
    picks = {"nsfnet": "160", "germany50": "320"}
    fair = {}
    for name in ["nsfnet", "germany50"]:
        L = picks[name]
        fair[name] = {}
        for pol in ["SP-FF", "KSP-FF", "Frag-Aware"]:
            rs = out[name][pol][L]
            fair[name][pol] = dict(
                by_rate={r: float(np.mean([x["by_rate"][r] for x in rs])) for r in rs[0]["by_rate"]},
                by_hops={h: float(np.mean([x["by_hops"].get(h, 0) for x in rs])) for h in rs[0]["by_hops"]})
    w = 0.26
    for j, pol in enumerate(["SP-FF", "KSP-FF", "Frag-Aware"]):
        br = fair["nsfnet"][pol]["by_rate"]
        ax[0].bar(np.arange(3) + (j - 1) * w, [br[k] for k in ("100", "200", "400")], w,
                  color=style.COLORS[pol], label=pol)
        bh = fair["germany50"][pol]["by_hops"]
        hs = sorted(int(h) for h in bh)
        hv = np.array([bh[str(h)] for h in hs])
        ok = hv > 0
        ax[1].plot(np.array(hs)[ok], hv[ok], color=style.COLORS[pol], marker=style.MARKERS[pol],
                   label=pol, mec="white")
    ax[0].set_xticks(range(3), ["100 Gb/s", "200 Gb/s", "400 Gb/s"])
    ax[0].set(ylabel="blocking probability", title=f"NSFNET at {picks['nsfnet']} Erlang: by bit rate")
    ax[0].legend()
    ax[1].set(xlabel="hops of the shortest path", ylabel="blocking probability", yscale="log",
              title=f"Germany50 at {picks['germany50']} Erlang: by distance")
    ax[1].legend()
    style.title(fig, "Who gets blocked: wide and long demands pay for everyone")
    fig.tight_layout()
    savefig(fig, "fig7_fairness")
    save_json("exp7_fairness", fair)


# 4 -------------------------------------------------------------------------
def exp_spectrum():
    net = net_of("nsfnet")
    L = 160
    out = {}
    snaps = {}
    for pol in ["KSP-FF", "Frag-Aware"]:
        r = run(net, POLICIES[pol], L, n_requests=20000, warmup=5000, seed=SEEDS[0], trace_every=250)
        tr = r.pop("trace")
        snaps[pol] = tr
        frag, util = [], []
        from eon.sim import free_blocks
        for i, occ in tr:
            fr = []
            for row in occ:
                free = ~row
                if free.sum():
                    _, ln = free_blocks(free)
                    fr.append(1 - ln.max() / free.sum())
            frag.append(float(np.mean(fr)))
            util.append(float(occ.mean()))
        out[pol] = dict(bp=r["bp"], step=[i for i, _ in tr], frag=frag, util=util)
    save_json("exp4_spectrum", out)
    np.savez_compressed(os.path.join(RES, "exp4_traces.npz"),
                        **{f"{p.replace('-', '_')}_{i}": occ for p in snaps for i, occ in snaps[p][::4]})
    style.apply()
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.38, wspace=0.12)
    k = len(snaps["KSP-FF"]) // 2
    for j, pol in enumerate(["KSP-FF", "Frag-Aware"]):
        ax = fig.add_subplot(gs[0, j])
        occ = snaps[pol][k][1]
        S_ = occ.shape[1]
        rgb = np.array(plt.get_cmap("turbo")(np.linspace(0.05, 0.95, S_)))[None, :, :3].repeat(occ.shape[0], 0)
        bg = np.array(matplotlib.colors.to_rgb(style.BG))
        rgb[~occ] = bg
        ax.imshow(rgb, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(0, occ.shape[0], 3))
        from eon.sim import free_blocks
        nfrag = sum(len(free_blocks(~row)[0]) for row in occ)
        ax.set(xlabel="frequency slot (12.5 GHz)", ylabel="link",
               title=f"{pol}: {occ.mean() * 100:.0f} % used, {nfrag} free fragments")
        ax.grid(False)
    ax = fig.add_subplot(gs[1, :])
    for pol in ["KSP-FF", "Frag-Aware"]:
        o = out[pol]
        x = np.array(o["step"]) / 1000
        ax.plot(x, o["frag"], color=style.COLORS[pol], label=f"{pol} fragmentation")
        ax.plot(x, o["util"], color=style.COLORS[pol], ls=":", label=f"{pol} utilisation")
    ax.axvline(snaps["KSP-FF"][k][0] / 1000, color=style.MUTED, lw=1)
    ax.set(xlabel="requests (thousands)", ylabel="mean over links",
           title="External fragmentation = 1 − largest free block / free slots")
    ax.legend(ncol=2, loc="lower right")
    style.title(fig, f"Inside the fibre: NSFNET at {L} Erlang, the same traffic for both policies",
                "each row is one link, each column one 12.5 GHz slot; coloured = occupied (colour = frequency), dark = free")
    savefig(fig, "fig4_spectrum")


# 5 -------------------------------------------------------------------------
def _abl(args):
    kind, val, pol, seed = args
    L = 160
    if kind == "format":
        fixed = None if val == "adaptive" else val
        r = run(net_of("nsfnet"), POLICIES[pol], L, N_REQ, WARM, seed=seed, fixed_format=fixed)
    elif kind == "k":
        r = run(net_of("nsfnet", k=val), POLICIES[pol], L, N_REQ, WARM, seed=seed)
    else:
        r = run(net_of("nsfnet"), make_fragmentation_aware(val), L, N_REQ, WARM, seed=seed)
    return kind, val, pol, seed, r["bp"], r["util"]


def exp_ablation(procs=1):
    jobs = [("format", v, p, s) for v in ["adaptive", "QPSK", "BPSK"] for p in ["KSP-FF", "Frag-Aware"] for s in SEEDS]
    jobs += [("k", v, p, s) for v in [1, 2, 3, 4, 5, 6, 8] for p in ["KSP-FF", "Frag-Aware"] for s in SEEDS]
    jobs += [("w", v, "Frag-Aware", s) for v in [0, 0.25, 0.5, 1, 2, 4, 8] for s in SEEDS]
    with Pool(procs) as pool:
        res = pool.map(_abl, jobs, chunksize=3)
    out = {}
    for kind, val, pol, seed, bp, ut in res:
        out.setdefault(kind, {}).setdefault(pol, {}).setdefault(str(val), []).append([bp, ut])
    save_json("exp5_ablation", out)
    fig_ablation()


def fig_ablation():
    out = load_json("exp5_ablation")
    style.apply()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    w = 0.36
    for j, pol in enumerate(["KSP-FF", "Frag-Aware"]):
        vals = ["adaptive", "QPSK", "BPSK"]
        m = [np.mean([x[0] for x in out["format"][pol][v]]) for v in vals]
        e = [ci95([x[0] for x in out["format"][pol][v]]) for v in vals]
        ax[0].bar(np.arange(3) + (j - 0.5) * w, m, w, yerr=e, color=style.COLORS[pol], label=pol,
                  capsize=3, ecolor="white")
        ks = sorted(int(k) for k in out["k"][pol])
        ax[1].errorbar(ks, [np.mean([x[0] for x in out["k"][pol][str(k)]]) for k in ks],
                       yerr=[ci95([x[0] for x in out["k"][pol][str(k)]]) for k in ks],
                       color=style.COLORS[pol], marker=style.MARKERS[pol], capsize=3, label=pol, mec="white")
    ax[0].set_xticks(range(3), ["distance-adaptive", "QPSK only", "BPSK only"])
    ax[0].set(ylabel="blocking probability", yscale="log", title="Modulation choice")
    ax[0].legend()
    ax[1].set(xlabel="k candidate paths", ylabel="blocking probability", yscale="log",
              title="How many paths to try")
    ax[1].legend()
    ws = sorted(float(v) for v in out["w"]["Frag-Aware"])
    key = lambda v: str(int(v)) if float(v).is_integer() else str(v)  # noqa: E731
    m = [np.mean([x[0] for x in out["w"]["Frag-Aware"][key(v)]]) for v in ws]
    e = [ci95([x[0] for x in out["w"]["Frag-Aware"][key(v)]]) for v in ws]
    ax[2].errorbar(range(len(ws)), m, yerr=e, color=style.COLORS["Frag-Aware"], marker="*", ms=12,
                   capsize=3, mec="white")
    ax[2].set_xticks(range(len(ws)), [f"{v:g}" for v in ws])
    ax[2].set(xlabel="weight w on cut fragments", ylabel="blocking probability", yscale="log",
              title="cost = slots × hops + w × cuts")
    style.title(fig, "Ablations on NSFNET at 160 Erlang",
                f"{len(SEEDS)} seeds × {N_REQ:,} requests each, bars = 95 % CI")
    fig.tight_layout()
    savefig(fig, "fig5_ablation")


# 6 -------------------------------------------------------------------------
def _drl_eval(args):
    import torch
    torch.set_num_threads(1)
    from eon.drl import ActorCritic, RMSAEnv, evaluate
    name, L, seed, mseed = args
    net = net_of(name)
    env = RMSAEnv(net, L)
    m = ActorCritic(env.obs_dim, env.n_actions)
    m.load_state_dict(torch.load(os.path.join(ROOT, "models", f"ppo_nsfnet_seed{mseed}.pt")))
    r = evaluate(net, m, L, n_requests=N_REQ, warmup=WARM, seed=seed)
    return name, L, seed, mseed, r["bp"]


def exp_drl(procs=1):
    mseeds = [1, 2]
    jobs = [(n, L, s, ms) for n in LOADS for L in LOADS[n] for s in SEEDS for ms in mseeds]
    with Pool(procs) as pool:
        res = pool.map(_drl_eval, jobs, chunksize=2)
    out = {}
    for name, L, seed, ms, bp in res:
        out.setdefault(name, {}).setdefault(str(ms), {}).setdefault(str(L), []).append(bp)
    curves = {ms: load_json(f"drl_curve_seed{ms}") for ms in mseeds}
    save_json("exp6_drl", dict(eval=out, curves=curves))
    fig_drl()


def fig_drl():
    d = load_json("exp6_drl")
    heur = load_json("exp3_blocking_raw")
    style.apply()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for ms, cv in d["curves"].items():
        cv = np.array(cv)
        ax[0].plot(cv[:, 0] / 1000, cv[:, 1], color=style.COLORS["DRL (PPO)"], alpha=0.35, lw=1)
        k = 5
        sm = np.convolve(cv[:, 1], np.ones(k) / k, mode="valid")
        ax[0].plot(cv[k - 1:, 0] / 1000, sm, color=style.COLORS["DRL (PPO)"], label=f"PPO seed {ms} (5-episode mean)",
                   ls="-" if ms == "1" else "--")
    ax[0].set(xlabel="training decisions (thousands)", ylabel="blocking per 20k-request episode",
              title="Training on NSFNET at 175 Erlang")
    ax[0].legend()
    for a, name in zip(ax[1:], ["nsfnet", "germany50"]):
        Ls = LOADS[name]
        for pol in ["KSP-FF", "Frag-Aware"]:
            m = [np.mean([r["bp"] for r in heur[name][pol][str(L)]]) for L in Ls]
            a.plot(Ls, m, color=style.COLORS[pol], marker=style.MARKERS[pol], label=pol, mec="white",
                   ms=9 if pol == "Frag-Aware" else 6)
        allbp = np.array([[np.mean(d["eval"][name][ms][str(L)]) for L in Ls] for ms in d["eval"][name]])
        a.plot(Ls, allbp.mean(0), color=style.COLORS["DRL (PPO)"], marker="P", ms=9, mec="white",
               label="DRL (PPO), mean of 2 agents")
        a.fill_between(Ls, allbp.min(0), allbp.max(0), color=style.COLORS["DRL (PPO)"], alpha=0.18)
        a.set_yscale("log")
        a.set(xlabel="offered load (Erlang)", ylabel="blocking probability",
              title=("NSFNET (trained here)" if name == "nsfnet" else "Germany50 (never seen in training)"))
        a.legend(loc="lower right")
    style.title(fig, "Does deep reinforcement learning beat the heuristics?",
                "same traffic (5 seeds) for every policy; shaded band = range of the two trained agents")
    fig.tight_layout()
    savefig(fig, "fig6_drl")


EXPS = {"physics": exp_physics, "maps": exp_maps, "blocking": exp_blocking, "spectrum": exp_spectrum,
        "ablation": exp_ablation, "drl": exp_drl, "fig_blocking": fig_blocking,
        "fig_ablation": fig_ablation, "fig_drl": fig_drl}

if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    names = sys.argv[1:] or ["physics", "maps", "blocking", "spectrum", "ablation", "drl"]
    procs = int(os.environ.get("PROCS", "2"))
    for n in names:
        t = time.time()
        print(n)
        f = EXPS[n]
        f(procs) if "procs" in f.__code__.co_varnames else f()
        print(f"  done in {time.time() - t:.0f} s")
