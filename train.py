#!/usr/bin/env python
# -*- encoding: utf-8 -*-

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
from contextlib import nullcontext
import os
import torch
import torch.distributed as dist


@internevo_monitor(feishu_alert=True, clean_run=True)
def main(args):
    # initialize model
    model = create_model()

    use_profile = False
    prof_ctx = nullcontext()
    prof_obj = None
    rank = dist.get_rank()
    if use_profile and rank == 0:
        logdir = "./profile_log"
        run_dir = os.path.join(logdir, f"job{os.environ.get('SLURM_JOB_ID', 'na')}_rank{rank}")
        os.makedirs(run_dir, exist_ok=True)
        torch.backends.cudnn.benchmark = True
        prof_obj = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_flops=False,
            with_modules=False,
            with_stack=False,
        )
        prof_ctx = prof_obj

    # initialize train dataloader
    train_dl, dataset_types = build_train_loader_with_data_type()

    # initialize validation dataloader
    val_dls = build_valid_loader_with_data_type()

    # build trainer
    merged_args = {**vars(args), "dataset_types": dataset_types}
    trainer = TrainerBuilder(model, train_dl, val_dls, **merged_args)

    # training
    with prof_ctx:
        trainer.fit()
    if use_profile and rank == 0:
        prof_obj.export_chrome_trace(os.path.join(run_dir, "trace_rank0.json"))

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    args = parse_args()

    # Initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    # Run the main function with parsed arguments
    main(args)
