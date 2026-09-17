"""Compatibility wrapper for CktGen's OFFICIAL training under torch 2.11.

Their code targets torch 1.13; torch >=2.4 removed the `verbose` kwarg from
ReduceLROnPlateau. This wrapper patches the scheduler class to accept and
ignore `verbose` (its only 1.13 effect was a print), then executes their
train.train_cktgen entry point verbatim with the official arguments passed
through. No CktGen source is modified. Documented in cktgen_audit.md.

Usage: python _cktgen_train_compat.py <cktgen_repo> [official train args...]
"""
import os
import runpy
import sys

repo = sys.argv[1]
os.chdir(repo)
sys.path.insert(0, repo)

import torch.optim.lr_scheduler as _ls

_orig = _ls.ReduceLROnPlateau.__init__


def _init(self, *a, verbose=None, **k):
    _orig(self, *a, **k)


_ls.ReduceLROnPlateau.__init__ = _init

sys.argv = ["train.train_cktgen"] + sys.argv[2:]
runpy.run_module("train.train_cktgen", run_name="__main__")
