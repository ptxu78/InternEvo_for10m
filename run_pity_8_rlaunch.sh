
# export PIP_INDEX_URL="http://mirrors.h.pjlab.org.cn/pypi/simple/"
# export PIP_EXTRA_INDEX_URL="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
# export PIP_TRUSTED_HOST="mirrors.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
# set -e: 错误及退出, 不会运行后续指令
# set -x: 打印执行轨迹, 所有命令的结果以及命令本身都会出现在日志中
set -e
export HOME="/mnt/shared-storage-user/ailab-sys/xuhaoran"
source ~/.bashrc

echo "/usr/local/nvidia/lib64" | sudo tee -a /etc/ld.so.conf && sudo ldconfig


export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
cd /mnt/shared-storage-user/ailab-sys/xuhaoran/InternEvo_for10m

# # rjob
# export MASTER_ADDR=${MASTER_ADDR}
# export GPUS_PER_NODE=$PROC_PER_NODE
# export MASTER_PORT=6001
# export NNODES=$NODE_COUNT
# export NODE_RANK=$NODE_RANK
# export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
# echo $WORLD_SIZE

# # rlaunch 8 rank
# # 单节点 8 卡固定这样写就行（再也不用管 slurm 变量了）
export NNODES=1                   # 节点数 = 1
export NODE_RANK=0                # 当前节点 rank 永远是 0
export GPUS_PER_NODE=8           # 每张机器 8 卡
export WORLD_SIZE=2               # 总卡数 = 1×8
export MASTER_ADDR="127.0.0.1"    # 单节点用 localhost 就够
export MASTER_PORT=29500          # 随便一个没被占用的端口，29500 是 torch 官方推荐

set +e  # 让后续指令即使出错也不会终端后续执行

CFG_SEQ_LEN=$((1024*256)) \
CFG_TOTAL_STEPS= \
CFG_PP_SIZE=1 \
CFG_HEAD_SIZE=1 \
CFG_CONTEXT_SIZE=8 \
CFG_WINDOW_SIZE=1 \
SEQ_K=$((CFG_SEQ_LEN/1024)) \
LOG_NAME="1111seq${SEQ_K}k_steps${CFG_TOTAL_STEPS}_pp${CFG_PP_SIZE}_h${CFG_HEAD_SIZE}_c${CFG_CONTEXT_SIZE}_ws${CFG_WINDOW_SIZE}" \
export CFG_SEQ_LEN CFG_TOTAL_STEPS CFG_PP_SIZE CFG_HEAD_SIZE CFG_CONTEXT_SIZE LOG_NAME CFG_WINDOW_SIZE; \
torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR train.py \
   --config configs/test_to10m_pity/7Binternlm2_seq_step_pp_hp_cp.py  \
   --launcher torch \
   2>&1 | tee "log/${LOG_NAME}.log"

