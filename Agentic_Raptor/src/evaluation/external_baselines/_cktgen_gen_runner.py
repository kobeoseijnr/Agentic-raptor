"""CktGen generation child-runner (topology-baseline experiment, 2026-08-29).

Runs CktGen's OWN auto-design flow (model + TPE multi-armed bandit + learned
evaluator) per mapped benchmark constraint, entirely with their code imported
verbatim from the frozen checkout. The only additions are OBSERVATION HOOKS:
module-level wrappers around spec_cond_gen / surrogate_model that record every
generated circuit and its surrogate prediction so the method's own ranking
(surrogate FoM, the signal that drives their search) can be exported. No
algorithmic behavior is changed; wrapped functions are called unmodified.

Usage: python _cktgen_gen_runner.py <cktgen_repo> <seed> <constraints.json> <out.json>
constraints.json: [{"spec_index": i, "cons": [g, b, p]}, ...]
"""
import json
import os
import sys
import time

REPO, SEED, CONS_JSON, OUT_JSON = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
TOP_K = 5

os.chdir(REPO)
sys.path.insert(0, REPO)
sys.argv = [
    "test_auto_design",
    "--data_name", "ckt_bench_101",
    "--data_fold_name", "CktBench101",
    "--out_dir", "./output/agbench",
    "--exp_name", f"agbench_s{SEED}",
    "--pretrained_eval_resume_pth", "./checkpoints/evaluator/evaluator_101.pth",
    "--resume_pth", "./checkpoints/cktgen/cktgen_cond_gen_101.pth",
]

import numpy as np
import random
import torch

from options.training import parser
import utils.paths as utils_paths
from utils.checkpoint import load_model_checkpoint
from dataset.get_datasets import get_datasets
import evaluation.auto_design as AD
import optuna

args = parser()
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
utils_paths.setup_paths(args)
os.makedirs(args["out_dir"], exist_ok=True)

datasets = get_datasets(args)
model = load_model_checkpoint(args["resume_pth"], map_location="cpu").to(args["device"])
model.eval()
surrogate = torch.load(args["pretrained_eval_resume_pth"],
                       map_location="cpu").to(args["device"])
surrogate.eval()

candidates = AD.get_all_specification_candidates(datasets)

# ---- observation hooks (record only; delegate verbatim) ---------------------
RECORD = []
_orig_gen = AD.spec_cond_gen
_orig_sur = AD.surrogate_model
_last_batch = []


def _gen_hook(a, m, batch):
    global _last_batch
    out = _orig_gen(a, m, batch)
    _last_batch = out
    return out


def _sur_hook(a, s, ckts):
    preds = _orig_sur(a, s, ckts)
    if ckts is _last_batch:  # record only the search batches, not re-scores
        for i, g in enumerate(ckts):
            RECORD.append((g, float(preds["gain"][i]), float(preds["bw"][i]),
                           float(preds["pm"][i]), float(preds["fom"][i])))
    return preds


AD.spec_cond_gen = _gen_hook
AD.surrogate_model = _sur_hook


def _num(x):
    if isinstance(x, torch.Tensor):
        return float(x.item()) if x.numel() == 1 else [float(v) for v in x.flatten()]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def ser_graph(g):
    vs = []
    for v in g.vs:
        a = v.attributes()
        vs.append({k: _num(a[k])
                   for k in ("type", "path", "r", "c", "gm") if k in a})
    return {"n": g.vcount(), "vs": vs,
            "edges": [[e.source, e.target] for e in g.es]}


results = []
for item in json.load(open(CONS_JSON, encoding="utf-8")):
    cons = tuple(item["cons"])
    RECORD.clear()
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    vc = AD.get_valid_candidates(candidates, cons)
    row = {"spec_index": item["spec_index"], "cons": list(cons),
           "n_valid_candidates": len(vc)}
    if not vc:
        row.update({"status": "no_valid_candidates", "candidates": [],
                    "generation_time_s": round(time.time() - t0, 2)})
        results.append(row)
        print(f"[s{SEED} spec{item['spec_index']}] cons={cons} NO CANDIDATES",
              flush=True)
        continue
    objective = AD.Objective(args=args, model=model, surrogate=surrogate,
                             valid_candidates=vc, constraint=cons,
                             optimize_sample_times=10)
    sampler = optuna.samplers.TPESampler(multivariate=True)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=len(vc) * 2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # method's own ranking: surrogate FoM descending over every circuit its
    # search generated (dedup by canonical serialization, keep best rank)
    ranked = sorted(RECORD, key=lambda r: -r[4])
    seen, top = set(), []
    for g, gain, bw, pm, fom in ranked:
        key = json.dumps(ser_graph(g), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        top.append({"graph": ser_graph(g), "pred_gain": gain, "pred_bw": bw,
                    "pred_pm": pm, "pred_fom": fom,
                    "their_valid": bool(AD.is_valid_Circuit(g))})
        if len(top) >= TOP_K:
            break
    row.update({"status": "ok", "n_generated": len(RECORD),
                "best_fom_surrogate": float(objective.best_fom),
                "candidates": top,
                "generation_time_s": round(time.time() - t0, 2)})
    results.append(row)
    print(f"[s{SEED} spec{item['spec_index']}] cons={cons} vc={len(vc)} "
          f"gen={len(RECORD)} t={row['generation_time_s']}s", flush=True)

json.dump({"seed": SEED, "results": results}, open(OUT_JSON, "w", encoding="utf-8"))
print("DONE", OUT_JSON, flush=True)
