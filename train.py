#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import logging
import os
import pickle
import time
from contextlib import contextmanager, ExitStack
from functools import partial
from pathlib import Path

import torch
import torch_npu
from torch import distributed as dist

from internlm.accelerator import get_accelerator, AcceleratorType
from internlm.core.context import global_context as gpc
from internlm.core.trainer_builder import TrainerBuilder
from internlm.data import (
    build_train_loader_with_data_type,
    build_valid_loader_with_data_type,
)
from internlm.initialize import initialize_distributed_env
from internlm.model.builder import create_model
from internlm.monitor import internevo_monitor
from internlm.utils.common import parse_args

logger = logging.getLogger(__name__)

class MemoryProfiler:
    def __init__(self, profile_dir: Path):
        torch.npu.memory._record_memory_history(
            max_entries=100_000,
            stacks="python",        # 关键:抓 Python 栈而不是 C++
            context="all",          # 同时记录 alloc 和 free 上下文
        )
        self.profile_dir = profile_dir
        # remember dirs already present so step() can pick whatever NEW
        # *_ascend_pt dir profiling_time creates — robust to mtime changes
        # from manual mv / cp into old dirs during the run
        self._preexisting = set(profile_dir.glob("*_ascend_pt")) if profile_dir.exists() else set()

    def step(self, exit_ctx: bool = False):
        if dist.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        new_dirs = sorted(
            set(self.profile_dir.glob("*_ascend_pt")) - self._preexisting,
            key=lambda p: p.name,   # name carries launch timestamp
        )
        if new_dirs:
            output_dir = new_dirs[-1]
            if exit_ctx:
                output_dir = output_dir.with_name(output_dir.name + "_exit")
        else:
            suffix = "_exit" if exit_ctx else ""
            output_dir = self.profile_dir / f"memory_snapshot_{time.strftime('%Y%m%d%H%M%S')}{suffix}"

        output_dir.mkdir(exist_ok=True, parents=True)
        snapshot_path = output_dir / f"rank{rank}_memory_snapshot.pickle"
        logger.info(f"Dumping memory snapshot to {snapshot_path}")
        begin = time.monotonic()
        with open(snapshot_path, "wb") as output:
            pickle.dump(torch.npu.memory._snapshot(), output)  # type: ignore
        logger.info(f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds")


@contextmanager
def profiling_memory(profile_dir: Path):
    profiler = MemoryProfiler(profile_dir)
    yield
    try:
        profiler.step(exit_ctx=False)
    except torch.OutOfMemoryError:
        profiler.step(exit_ctx=True)


@contextmanager
def profiling_time(profile_dir: Path):
    # experimental_config = torch_npu.profiler._ExperimentalConfig(
    #     export_type=[
    #         torch_npu.profiler.ExportType.Text,
    #         torch_npu.profiler.ExportType.Db
    #         ],
    #     profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    #     msprof_tx=False,
    #     aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    #     l2_cache=False,
    #     op_attr=False,
    #     data_simplification=False,
    #     record_op_args=False,
    #     gc_detect_threshold=None
    # )

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )

    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        # schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        experimental_config=experimental_config,
    ) as prof:
        yield

        prof.step()

@contextmanager
def _maybe_profile(use_profile, rank, dir):
    if use_profile and rank == 0:
        with ExitStack() as stack:
            # Enter memory first so it exits LAST — after profiling_time has
            # written the *_ascend_pt dir, which MemoryProfiler globs into.
            stack.enter_context(profiling_memory(Path(dir)))
            stack.enter_context(profiling_time(Path(dir)))

            yield
    else:
        yield

@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    # initialize model
    model = create_model()

    use_profile = False
    prof_obj = None
    rank = dist.get_rank()
    logdir = "./profile_log"
    run_dir = os.path.join(logdir, f"job{os.environ.get('SLURM_JOB_ID', os.environ.get('JOB_LAUNCH_TIME', 'na'))}_rank{rank}")
    if use_profile and rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # build trainer
    merged_args = {**vars(args), "dataset_types": dataset_types}
    trainer = TrainerBuilder(model, train_dl, val_dls, **merged_args)

    # training
    with _maybe_profile(use_profile, rank, run_dir):
        trainer.fit()

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)
