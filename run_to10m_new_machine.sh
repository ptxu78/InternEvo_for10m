#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV:-internevo}"
fi

export INTERNLM_ACCELERATOR="${INTERNLM_ACCELERATOR:-npu}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

CONFIG="${CONFIG:-configs/test_to10m_pity/7Binternlm2_seq_step_pp_hp_cp.py}"

CFG_SEQ_LEN="${CFG_SEQ_LEN:-$((1024 * 64))}"
CFG_TOTAL_STEPS="${CFG_TOTAL_STEPS:-3}"
CFG_PP_SIZE="${CFG_PP_SIZE:-1}"
CFG_HEAD_SIZE="${CFG_HEAD_SIZE:-2}"
CFG_CONTEXT_SIZE="${CFG_CONTEXT_SIZE:-8}"
CFG_WINDOW_SIZE="${CFG_WINDOW_SIZE:-1}"
CFG_USE_FLASH_ATTN="${CFG_USE_FLASH_ATTN:-True}"
CFG_NPU_FIXEDLEN_FLASH_2D="${CFG_NPU_FIXEDLEN_FLASH_2D:-True}"
CFG_OVERLAP_SYNC_GRAD="${CFG_OVERLAP_SYNC_GRAD:-False}"
CFG_OVERLAP_SYNC_PARAM="${CFG_OVERLAP_SYNC_PARAM:-False}"
CFG_FIXED_RANDOM_DATASET_SEQLEN="${CFG_FIXED_RANDOM_DATASET_SEQLEN:-True}"
CFG_EMPTY_CACHE_AND_DIAG_INTERVAL="${CFG_EMPTY_CACHE_AND_DIAG_INTERVAL:-200}"

SEQ_K=$((CFG_SEQ_LEN / 1024))
FLASH_TAG=""
if [ "${CFG_USE_FLASH_ATTN}" != "False" ] || [ "${CFG_NPU_FIXEDLEN_FLASH_2D}" != "False" ]; then
  FLASH_TAG="_fa${CFG_USE_FLASH_ATTN}_npu2d${CFG_NPU_FIXEDLEN_FLASH_2D}"
fi
_LAUNCH_TIMESTAMP=`date +%Y%m%d%H%M%S`
LOG_NAME="${LOG_NAME:-${_LAUNCH_TIMESTAMP}_seq${SEQ_K}k_steps${CFG_TOTAL_STEPS}_pp${CFG_PP_SIZE}_h${CFG_HEAD_SIZE}_c${CFG_CONTEXT_SIZE}_ws${CFG_WINDOW_SIZE}${FLASH_TAG}}"

export CFG_SEQ_LEN CFG_TOTAL_STEPS CFG_PP_SIZE CFG_HEAD_SIZE CFG_CONTEXT_SIZE CFG_WINDOW_SIZE CFG_USE_FLASH_ATTN CFG_NPU_FIXEDLEN_FLASH_2D CFG_OVERLAP_SYNC_GRAD CFG_OVERLAP_SYNC_PARAM CFG_FIXED_RANDOM_DATASET_SEQLEN CFG_EMPTY_CACHE_AND_DIAG_INTERVAL LOG_NAME

mkdir -p log

# ---------------------------------------------------------------------------
# 进程清理(防僵尸 / 防 EJ0003 端口占用)
#   - 启动前:清掉上一次崩溃残留的 rank 进程,否则它们还占着 HCCL/master 端口,
#     新任务会报 Communication_Error_Bind_IP_Port(EJ0003)。
#   - 收到中断信号 / 本次崩溃后:把没退干净的 worker 一并收掉,避免孤儿堆积。
#   - 仅作用于本 pod 的 PID 命名空间,不会误伤同节点其他 pod。
#   - 注意:卡在 NPU 驱动里的 D 态(uninterruptible)进程 kill 不掉,
#     需复位 NPU 设备或重启节点,这不是脚本能处理的。
#   - 如需跳过启动前清理:export SKIP_PREFLIGHT_CLEANUP=1
# ---------------------------------------------------------------------------
cleanup_stragglers() {
  pkill -9 -f 'torchrun'  2>/dev/null || true
  pkill -9 -f 'train\.py' 2>/dev/null || true
}
trap 'echo "[run] 收到中断信号,清理子进程后退出..."; cleanup_stragglers; exit 130' INT TERM

if [ "${SKIP_PREFLIGHT_CLEANUP:-0}" != "1" ]; then
  echo "[run] 启动前清理上一次残留的 torchrun/train.py 进程..."
  cleanup_stragglers
  sleep 2   # 给 OS 一点时间释放被占用的端口
fi

echo "CONFIG=${CONFIG}"
echo "LOG=log/${LOG_NAME}.log"
echo "NNODES=${NNODES} NODE_RANK=${NODE_RANK} GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "CFG_SEQ_LEN=${CFG_SEQ_LEN} CFG_TOTAL_STEPS=${CFG_TOTAL_STEPS}"
echo "CFG_PP_SIZE=${CFG_PP_SIZE} CFG_HEAD_SIZE=${CFG_HEAD_SIZE} CFG_CONTEXT_SIZE=${CFG_CONTEXT_SIZE} CFG_WINDOW_SIZE=${CFG_WINDOW_SIZE}"
echo "CFG_USE_FLASH_ATTN=${CFG_USE_FLASH_ATTN} CFG_NPU_FIXEDLEN_FLASH_2D=${CFG_NPU_FIXEDLEN_FLASH_2D}"
echo "CFG_OVERLAP_SYNC_GRAD=${CFG_OVERLAP_SYNC_GRAD} CFG_OVERLAP_SYNC_PARAM=${CFG_OVERLAP_SYNC_PARAM} CFG_FIXED_RANDOM_DATASET_SEQLEN=${CFG_FIXED_RANDOM_DATASET_SEQLEN}"
echo "CFG_EMPTY_CACHE_AND_DIAG_INTERVAL=${CFG_EMPTY_CACHE_AND_DIAG_INTERVAL}"

torchrun \
  --nnodes "${NNODES}" \
  --nproc-per-node "${GPUS_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --max-restarts 0 \
  train.py \
  --config "${CONFIG}" \
  --launcher torch \
  2>&1 | tee "log/${LOG_NAME}.log"

rc="${PIPESTATUS[0]}"

# 收尾:本次若非正常退出,torchrun 可能留下没收干净的 worker,补一刀
if [ "${rc}" -ne 0 ]; then
  echo "[run] torchrun 退出码 ${rc},清理可能残留的 worker 进程..."
  cleanup_stragglers
fi

exit "${rc}"
