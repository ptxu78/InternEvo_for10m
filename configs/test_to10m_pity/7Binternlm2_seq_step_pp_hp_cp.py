import os
import ast

def _get_env(name, default):
    """
    从环境变量读取覆盖值：
    - 不存在就用 default
    - 存在就尝试用 ast.literal_eval 解析（支持数字/True/False/字符串/列表等）
    """
    v = os.environ.get(name, None)
    if v is None or v == "":
        return default
    try:
        return ast.literal_eval(v)
    except Exception:
        # 如果不是合法 literal（比如纯字符串没加引号），就原样用
        return v
JOB_NAME = "7b_internlm2_train_pityto10m"
model_type = "INTERNLM2"
DO_ALERT = False

VOCAB_SIZE = _get_env("CFG_VOCAB_SIZE", 92544)
SEQ_LEN = _get_env("CFG_SEQ_LEN", 1024 * 1024)
HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 3.5
NUM_LAYER = 32

MODEL_ONLY_FOLDER = "local:llm_ckpts/xxxx"
# Ckpt folder format:
# fs: 'local:/mnt/nfs/XXX'
SAVE_CKPT_FOLDER = "local:llm_ckpts"
LOAD_CKPT_FOLDER = "local:llm_ckpts/49"
LOAD_HF_CHECKPOINT = bool(_get_env("CFG_LOAD_HF_CHECKPOINT", False))
HF_CHECKPOINT_FOLDER = str(
    _get_env(
        "CFG_HF_CHECKPOINT_FOLDER",
        "local:/MindSpeed-LLM/InternEvo_for10m/llm_ckpts/hf/internlm2-7b",
    )
)

# boto3 Ckpt folder format:
# import os
# BOTO3_IP = os.environ["BOTO3_IP"] # boto3 bucket endpoint
# SAVE_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm"
# LOAD_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm/snapshot/1/"
CHECKPOINT_EVERY = 50
ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    # The official InternLM2-7B Hugging Face weights are loaded only when
    # CFG_LOAD_HF_CHECKPOINT=True. Keep it False for a scratch baseline.
    load_ckpt_info=(
        dict(path=HF_CHECKPOINT_FOLDER, content=("model",), ckpt_type="hf") if LOAD_HF_CHECKPOINT else None
    ),
    # 'auto_resume' is designed to automatically load the latest checkpoint from 'save_ckpt_folder' when encountering
    # training interruptions/hangs caused by hardware failures, using a scheduling system (such as k8s/slurm)
    # with an automatic restart mechanism upon training reboot.
    # Please be aware that if `auto_resume` is not set (its default value is True), it will not load the checkpoint
    # path specified in `load_ckpt_info` by default.
    # If you want to initialize your model weights from another model, you must set `auto_resume` to False.
    # If you want to train from scratch, please set `auto_resume` to False and 'load_ckpt_info' to None.
    auto_resume=False,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
)

TRAIN_FOLDER = _get_env("CFG_TRAIN_FOLDER", None)
VALID_FOLDER = _get_env("CFG_VALID_FOLDER", None)
data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    micro_num=_get_env("CFG_MICRO_NUM", 1),
    # packed_length = micro_bsz * SEQ_LEN
    micro_bsz=1,
    # defaults to the value of micro_num
    valid_micro_num=_get_env("CFG_VALID_MICRO_NUM", 4),
    # defaults to 0, means disable evaluate
    valid_every=_get_env("CFG_VALID_EVERY", 0),
    pack_sample_into_one=False,
    total_steps=_get_env("CFG_TOTAL_STEPS", 5),
    skip_batches=str(_get_env("CFG_SKIP_BATCHES", "")),
    # rampup_batch_size (str): A string with three space-separated integers representing the
    #       starting batch size, the increment, and the number of steps between
    #       each increment. For example, "192 24 8" means that the batch size (micro_num)
    #       starts at 192 and increases by 24 every 8 steps. Defaults to None.
    #       (IMPORTANT): The interval step size is 'micro_bsz'.
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=_get_env("CFG_MIN_LENGTH", 50),
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=_get_env("CFG_EMPTY_CACHE_AND_DIAG_INTERVAL", 200),
    empty_cache_before_backward=_get_env("CFG_EMPTY_CACHE_BEFORE_BACKWARD", False),
    empty_cache_before_optimizer=_get_env("CFG_EMPTY_CACHE_BEFORE_OPTIMIZER", False),
    empty_cache_before_param_broadcast=_get_env("CFG_EMPTY_CACHE_BEFORE_PARAM_BROADCAST", False),
    diag_outlier_ratio=1.1,
    use_packed_dataset=_get_env("CFG_USE_PACKED_DATASET", False),
    fixed_random_dataset_seqlen=_get_env("CFG_FIXED_RANDOM_DATASET_SEQLEN", True),
    random_dataset_num_samples=_get_env("CFG_RANDOM_DATASET_NUM_SAMPLES", 500),
    repeat_dataset=_get_env("CFG_REPEAT_DATASET", 1),
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=_get_env("CFG_OVERLAP_SYNC_GRAD", False),
    overlap_sync_param=_get_env("CFG_OVERLAP_SYNC_PARAM", False), # True
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=_get_env("CFG_CLIP_GRAD_NORM", 1.0),
)

# loss config (dict):
#     1. label_smoothing
#     2. op_type: cross_entropy operator type, we support five types for loss computing,
#                 including ["torch_naive", "apex_naive", "py_naive", "flash_vocab_parallel", "py_vocab_parallel"]
#                 default is "py_vocab_parallel".
#         "torch_naive": cross_entropy imported from torch, i.e. torch.nn.CrossEntropyLoss
#         "apex_naive": cross_entropy from apex
#         "py_naive": self-implemented cross_entropy
#         "flash_vocab_parallel": vocab parallel cross_entropy imported from flash_attn
#         "py_vocab_parallel": self-implemented vocab parallel cross_entropy

#         * op_types that ends with "naive" only support parallel_output=False;
#         * if in no-GPU env, only "torch_naive" and "py_vocab_parallel" are supported.
loss = dict(label_smoothing=0, op_type="py_vocab_parallel")

adam = dict(
    lr=_get_env("CFG_LR", 1e-4),
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=_get_env("CFG_WARMUP_RATIO", 0.01),
    eta_min=_get_env("CFG_ETA_MIN", 1e-5),
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

# cpu_offloading = dict(
#     enable=True,
#     num_layers=3,
# )
selective_checkpoint = False
# selective_checkpoint_offload = False

use_fp32_norm = False
USE_FLASH_ATTN = _get_env("CFG_USE_FLASH_ATTN", False)
npu_fixedlen_flash_2d = _get_env("CFG_NPU_FIXEDLEN_FLASH_2D", False)
model = dict(
    checkpoint=True,
    num_chunks=1,
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    no_bias=True,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    dtype="torch.bfloat16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    use_flash_attn=USE_FLASH_ATTN,
    # Match internlm/internlm2-7b config.json (rope_theta=1_000_000).
    # This is kept identical for checkpoint and scratch comparison runs.
    rope_base=1_000_000,
    # Whether the odd and even columns of the query and key in the model are normally interleaved.
    # If it's True, the model's odd and even columns are normally ordered; if it's False,
    # it means that the model has prematurely concatenated all odd columns and even columns in front
    # and back, in order to improve the RoPE's computational efficiency.
    # Example:
    # qk_interleaved = True: q[-1] = [q1,q2,q3,q4,q5,q6,...], k[-1] = [k1,k2,k3,k4,k5,k6,...]
    # qk_interleaved = False: q[-1] = [q1,q3,q5,...,q2,q4,q6,...], k[-1] = [k1,k3,k5,...,k2,k4,k6,...]
    qk_interleaved=False,
)

"""
zero1 parallel (dict):
    1. size: int
        * if size <= 0, the size of the zero process group is equal to the size of the dp process group,
            so parameters will be divided within the range of dp.
        * if size == 1, zero is not used, and all dp groups retain the full amount of model parameters.
        * if size > 1 and size <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
tensor parallel (dict):
    1. size: int, the size of tensor parallel.
    2. mode: str, the tensor parallel mode, should be in ['mtp', 'msp', 'fsp', 'isp'],
        defaults to 'mtp', means the pure megatron tensor parallel without sequence parallel.
        msp: megatron tensor parallel with sequence parallel, sequence parallel size = tensor parallel size.
        fsp: tensor parallel by flash-attn with sequence parallel, sequence parallel size = tensor parallel size.
        isp: customed intern sequence parallel without tensor parallel, can be used with weight parallel.
pipeline parallel (dict):
    1. size: int, the size of pipeline parallel.
    2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler,
        defaults to False.
    3. mode: str, the pipeline parallel mode, should be in ['1f1b', 'zbh1', 'zbv']. The defalut is 1f1b.
weight parallel (dict):
    1. size: int, the size of weight parallel.
    2. overlap: bool, enable/disable all_gather/reduce_scatter communication overlap, defaults to False.
"""

HEAD_SIZE = _get_env("CFG_HEAD_SIZE", 8)
CONTEXT_SIZE = _get_env("CFG_CONTEXT_SIZE", 4)
TP_SIZE = HEAD_SIZE * CONTEXT_SIZE

parallel = dict(
    zero1=dict(size=-1),
    tensor=dict(size=TP_SIZE, mode="isp"),
    pipeline=dict(size=_get_env("CFG_PP_SIZE", 1), interleaved_overlap=True),
    weight=dict(
        size=HEAD_SIZE * CONTEXT_SIZE,
        # ISP weight-prefetch overlap can leave outstanding all-gather state
        # when switching from training to validation on this NPU stack.
        overlap=_get_env("CFG_WEIGHT_OVERLAP", True),
        launch_allgather_before="wo",
        forward_overlap_per="layer",
    ),
    sequence_2D=dict(
        enable=True,
        head_size=HEAD_SIZE,
        context_size=CONTEXT_SIZE,
        window_size=_get_env("CFG_WINDOW_SIZE", 1),
        device_placement_strategy=dict(head_first=True, interleaved=False),
    ),
)


cudnn_deterministic = False
cudnn_benchmark = False

monitor = dict(
    # feishu alert configs
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
    tensorboard=dict(
        queue_max_length=10,
    ),
)

# metric_dtype can be "fp32" or other string
# only when set to "fp32" will use fp32 to calc in metrics
# metric_dtype = "fp32"

generation = dict(
    ckpt_folder="/path/to/saved/ckpt",
    output_folder="/path/to/save/generation",
    batch_size=1,
    eos_id=[2, 0],
    bos_id=1,
    max_length=100,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1,
    length_penalty=1.0,
)
