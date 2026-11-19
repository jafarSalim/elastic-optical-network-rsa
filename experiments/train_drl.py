"""Train the PPO agent on NSFNET at 175 Erlang (two seeds) and save the models.

    python experiments/train_drl.py            # about 25 minutes on one CPU core
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from eon.drl import train_ppo  # noqa: E402
from eon.topology import load  # noqa: E402

LOAD = 175
STEPS = 4096 * 120

if __name__ == "__main__":
    torch.set_num_threads(1)
    net = load("nsfnet")
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2]
    for seed in seeds:
        print(f"seed {seed}", flush=True)
        model, curve = train_ppo(net, LOAD, total_steps=STEPS, seed=seed,
                                 log=lambda m: print(m, flush=True))
        torch.save(model.state_dict(), os.path.join(ROOT, "models", f"ppo_nsfnet_seed{seed}.pt"))
        with open(os.path.join(ROOT, "results", f"drl_curve_seed{seed}.json"), "w") as f:
            json.dump(curve, f)
