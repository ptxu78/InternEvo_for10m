rjob delete rjob-32rank-loongtrain-opt-act1-test
rjob submit \
--name=rjob-32rank-loongtrain-opt-act1-test  \
--gpu=8 \
--memory=1600000 \
--cpu=128 \
--charged-group=sys_gpu \
--private-machine=group \
--mount=gpfs://gpfs1/ailab-sys:/mnt/shared-storage-user/ailab-sys \
--image=registry.h.pjlab.org.cn/library/ml-base:22.04-pjlab \
-P 4 \
--host-network=true \
-e DISTRIBUTED_JOB=true \
-- bash -exc /mnt/shared-storage-user/ailab-sys/xuhaoran/InternEvo_for10m/run_pity_32.sh
