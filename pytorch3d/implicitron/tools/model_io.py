# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import shutil
import tempfile
from typing import Optional

import torch


logger = logging.getLogger(__name__)


def load_stats(flstats):
    from pytorch3d.implicitron.tools.stats import Stats

    return None if not os.path.isfile(flstats) else Stats.load(flstats)


def get_model_path(fl) -> str:
    fl = os.path.splitext(fl)[0]
    return f"{fl}.pth"


def get_optimizer_path(fl) -> str:
    fl = os.path.splitext(fl)[0]
    return f"{fl}_opt.pth"


def get_stats_path(fl, eval_results: bool = False) -> str:
    fl = os.path.splitext(fl)[0]
    if eval_results:
        for postfix in ("_2", ""):
            flstats = os.path.join(os.path.dirname(fl), f"stats_test{postfix}.jgz")
            if os.path.isfile(flstats):
                break
    else:
        flstats = f"{fl}_stats.jgz"
    # pyre-fixme[61]: `flstats` is undefined, or not always defined.
    return flstats


def safe_save_model(model, stats, fl, optimizer=None, cfg=None) -> None:
    """
    This functions stores model files safely so that no model files exist on the
    file system in case the saving procedure gets interrupted.

    This is done first by saving the model files to a temporary directory followed
    by (atomic) moves to the target location. Note, that this can still result
    in a corrupt set of model files in case interruption happens while performing
    the moves. It is however quite improbable that a crash would occur right at
    this time.
    """
    logger.info(f"saving model files safely to {fl}")
    # first store everything to a tmpdir
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfl = os.path.join(tmpdir, os.path.split(fl)[-1])
        stored_tmp_fls = save_model(model, stats, tmpfl, optimizer=optimizer, cfg=cfg)
        tgt_fls = [
            (
                os.path.join(os.path.split(fl)[0], os.path.split(tmpfl)[-1])
                if (tmpfl is not None)
                else None
            )
            for tmpfl in stored_tmp_fls
        ]
        # then move from the tmpdir to the right location
        for tmpfl, tgt_fl in zip(stored_tmp_fls, tgt_fls):
            if tgt_fl is None:
                continue
            shutil.move(tmpfl, tgt_fl)


def save_model(model, stats, fl, optimizer=None, cfg=None):
    flstats = get_stats_path(fl)
    flmodel = get_model_path(fl)
    logger.info(f"saving model to {flmodel}")
    torch.save(model.state_dict(), flmodel)
    flopt = None
    if optimizer is not None:
        flopt = get_optimizer_path(fl)
        logger.info(f"saving optimizer to {flopt}")
        torch.save(optimizer.state_dict(), flopt)
    logger.info(f"saving model stats to {flstats}")
    stats.save(flstats)

    return flstats, flmodel, flopt


def load_model(fl, map_location: Optional[dict]):
    flstats = get_stats_path(fl)
    flmodel = get_model_path(fl)
    flopt = get_optimizer_path(fl)
    model_state_dict = torch.load(flmodel, map_location=map_location)
    stats = load_stats(flstats)
    if os.path.isfile(flopt):
        optimizer = torch.load(flopt, map_location=map_location)
    else:
        optimizer = None

    return model_state_dict, stats, optimizer


def parse_epoch_from_model_path(model_path) -> int:
    return int(
        os.path.split(model_path)[-1].replace(".pth", "").replace("model_epoch_", "")
    )


def get_checkpoint(exp_dir, epoch):
    return os.path.join(exp_dir, "model_epoch_%08d.pth" % epoch)


def find_last_checkpoint(
    exp_dir, any_path: bool = False, all_checkpoints: bool = False
):
    exts = [".pth", "_stats.jgz", "_opt.pth"] if any_path else [".pth"]
    for ext in exts:
        fls = sorted(
            glob.glob(
                os.path.join(glob.escape(exp_dir), "model_epoch_" + "[0-9]" * 8 + ext)
            )
        )
        if len(fls) > 0:
            break
    # pyre-fixme[61]: `fls` is undefined, or not always defined.
    if len(fls) == 0:
        return None
    else:
        return (
            [f"{f[:-len(ext)]}.pth" for f in fls]
            if all_checkpoints
            else f"{fls[-1][:-len(ext)]}.pth"
        )


def purge_epoch(exp_dir, epoch) -> None:
    model_path = get_checkpoint(exp_dir, epoch)

    for file_path in [
        model_path,
        get_optimizer_path(model_path),
        get_stats_path(model_path),
    ]:
        if os.path.isfile(file_path):
            logger.info(f"deleting {file_path}")
            os.remove(file_path)
