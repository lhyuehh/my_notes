#!/bin/bash
set -ex # 打印执行的命令，方便调试

# --- 函数区 ---

# 函数：寻找一个空闲的 TCP 端口
find_free_port() {
  python -c '
import socket
from contextlib import closing
with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])'
}

# 函数：检查指定的端口是否被占用
is_port_free() {
  local port="$1"
  if sudo netstat -lnt | grep -q ":$port\b"; then return 1; else return 0; fi
}

# 函数：获取一个可用的端口
get_available_port() {
  local preferred_port="${ARNOLD_WORKER_0_PORT}"
  if [[ -n "$preferred_port" ]]; then
    echo "Platform assigned port: $preferred_port. Checking..." >&2
    if is_port_free "$preferred_port"; then
      echo "Port $preferred_port is free. Using it." >&2
      echo "$preferred_port"
      return
    else
      echo "Warning: Port $preferred_port is in use! Finding a new one." >&2
    fi
  fi
  local free_port=$(find_free_port)
  echo "Automatically found free port: $free_port" >&2
  echo "$free_port"
}

# 函数：【核心】将可能为 IPv6 的地址转换为对应的 IPv4 地址
get_ipv4_equivalent() {
  local input_addr="$1"
  
  # 检查输入地址是否为 IPv6 (包含冒号)
  if [[ "$input_addr" == *":"* ]]; then
    echo "Input address '$input_addr' is IPv6. Attempting to find its IPv4 equivalent..." >&2
    # 从 `hostname -I` 的输出中查找第一个合法的 IPv4 地址
    local ipv4_addr=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/) {print $i; exit}}')
    
    if [[ -n "$ipv4_addr" ]]; then
      echo "Found corresponding IPv4 address: $ipv4_addr" >&2
      # 输出找到的 IPv4 地址
      echo "$ipv4_addr"
    else
      echo "Warning: Could not find an IPv4 equivalent for '$input_addr'. Using original address." >&2
      # 如果找不到，作为后备，返回原始地址（虽然可能失败）
      echo "$input_addr"
    fi
  else
    # 如果输入地址已经是 IPv4，直接输出
    echo "Input address '$input_addr' is already IPv4. Using it directly." >&2
    echo "$input_addr"
  fi
}


# --- 1. 获取并格式化分布式训练参数 ---

# --- 地址处理 ---
# 从环境变量获取原始地址
ADDR_FROM_ENV="${ARNOLD_WORKER_0_HOST:-"127.0.0.1"}"

# 【转换】调用转换器函数，获取最终要使用的地址
# 这一步会把 IPv6 转换成 IPv4
FINAL_MASTER_ADDR=$(get_ipv4_equivalent "$ADDR_FROM_ENV")

# --- 端口处理 ---
MASTER_PORT=$(get_available_port)

echo "Final Master Address for torchrun: $FINAL_MASTER_ADDR"
echo "Final Master Port for torchrun: $MASTER_PORT"


init_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY # 清除代理环境变量
  # 安装或升级各种 Python 库
  pip install "deepspeed>=0.10.0,<=0.16.9"
  # --- 在这里添加你的 LLaMA-Factory 安装和 metrics 依赖 ---
  # 获取脚本所在的目录
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  # LLaMA-Factory 的根目录是脚本目录的上一级
  LLAMA_FACTORY_ROOT="$(dirname "$SCRIPT_DIR")" 
  echo "Changing directory to LLaMA-Factory root: $LLAMA_FACTORY_ROOT for pip installs."
  cd "$LLAMA_FACTORY_ROOT"
  echo "Installing LLaMA-Factory in editable mode..."
  pip install -e . || true
  echo "Installing dependencies from requirements/metrics.txt..."
  pip install -r requirements/metrics.txt || true # 安装 requirements.txt 中的依赖，失败不中断
  # 安装完成后，切换回脚本所在的目录
  echo "Changing back to script directory: $SCRIPT_DIR."
  cd "$SCRIPT_DIR"
  # ------------------------------------------------------------------

  nvidia-smi # 打印 GPU 信息，确认 GPU 可用
  #llamafactory-cli version #检查llamafactory是否安装成功


  # 获取分布式训练相关的环境变量，通常来自集群调度系统（如 Arnold）
  WORKER_NUM=${ARNOLD_WORKER_NUM:-1} # 总节点数，默认为 1
  WORKER_GPU=${ARNOLD_WORKER_GPU:-1} # 每节点 GPU 数，默认为 1
  NPROC_PER_NODE="$WORKER_GPU" # 每节点进程数（即 GPU 数）
  
  # 组装 torchrun 的分布式参数
  DIST_ARGS=(
    --nnodes="$WORKER_NUM"
    --node_rank="${ARNOLD_ID:-0}" # 当前节点在集群中的排名，默认为 0
    --nproc_per_node="$NPROC_PER_NODE"
    --master_addr="$FINAL_MASTER_ADDR"
    --master_port="$MASTER_PORT"
  )
}
# --- 2. 准备并执行 torchrun 命令 ---

# 默认模型路径
MODEL_NAME_OR_PATH="/opt/tiger/ttls_rel_llama_factory/ttls_poi_cpt/model/Qwen3-0.6B"
# 默认输出目录（HDFS 挂载目录）
FINAL_OUTPUT_DIR="/mnt/hdfs/qwen3_0.6B/arnold_output"
# 临时本地输出目录
TEMP_OUTPUT_DIR="./saved_models"
# WandB 默认配置
WANDB_PROJECT="ttls_poi_cpt"
WANDB_NAME="qwen3_train"

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --model_name_or_path) MODEL_NAME_OR_PATH="$2"; echo "Parsed model_name_or_path: ${MODEL_NAME_OR_PATH}"; shift; shift ;;
      --output_dir) FINAL_OUTPUT_DIR="$2"; echo "Parsed output_dir: ${FINAL_OUTPUT_DIR}"; shift; shift ;;
      --wandb_project) WANDB_PROJECT="$2"; echo "Parsed wandb_project: ${WANDB_PROJECT}"; shift; shift ;;
      --wandb_name) WANDB_NAME="$2"; echo "Parsed wandb_name: ${WANDB_NAME}"; shift; shift ;;
      *) # 忽略其他参数或可按需扩展
        shift
        ;;
    esac
  done
}

prepare_inputs() {
  # 设置 WandB 环境变量
  export WANDB_PROJECT="${WANDB_PROJECT}"
  # 如果在 Arnold 环境中，增加 Trial ID 区分
  if [[ -n "$ARNOLD_TRIAL_ID" ]]; then
    WANDB_NAME="${WANDB_NAME}_${ARNOLD_TRIAL_ID}"
  else
    WANDB_NAME="${WANDB_NAME}_$(date +%Y%m%d_%H%M%S)"
  fi
  export WANDB_NAME="${WANDB_NAME}"

  # 更新输出目录，增加 wandb_name 子目录
  FINAL_OUTPUT_DIR="${FINAL_OUTPUT_DIR}/${WANDB_NAME}"
  TEMP_OUTPUT_DIR="${TEMP_OUTPUT_DIR}/${WANDB_NAME}"

  # 获取脚本所在的目录
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  
  # 去掉 MODEL_NAME_OR_PATH 末尾可能存在的斜杠
  MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH%/}"

  local_models_root="${SCRIPT_DIR}/model" # 在脚本当前目录下创建 model 文件夹
  mkdir -p "$local_models_root"

  if [[ "$MODEL_NAME_OR_PATH" == hdfs://* ]]; then # 如果是 HDFS 路径
    local_model_name="$(basename "$MODEL_NAME_OR_PATH")"
    LOCAL_MODEL_PATH="${local_models_root}/${local_model_name}"
    echo "Downloading model from HDFS: $MODEL_NAME_OR_PATH to $LOCAL_MODEL_PATH"
    if [[ -e "$LOCAL_MODEL_PATH" ]]; then rm -rf "$LOCAL_MODEL_PATH"; fi # 如果本地已存在，则删除
    hdfs dfs -copyToLocal "$MODEL_NAME_OR_PATH" "$local_models_root" || hdfs dfs -get "$MODEL_NAME_OR_PATH" "$local_models_root"
    
    # 验证模型文件夹是否存在且不为空
    if [ ! -d "$LOCAL_MODEL_PATH" ] || [ -z "$(ls -A "$LOCAL_MODEL_PATH" 2>/dev/null)" ]; then
      echo "Error: Failed to download model from HDFS or model directory is empty: $LOCAL_MODEL_PATH"
      exit 1
    fi
  else # 否则认为是本地路径
    LOCAL_MODEL_PATH="$MODEL_NAME_OR_PATH"
    echo "Using local model path: $LOCAL_MODEL_PATH"
  fi

  # 准备输出目录
  mkdir -p "$TEMP_OUTPUT_DIR"
  mkdir -p "$FINAL_OUTPUT_DIR"
}

# 异步同步函数
start_sync_output() {
  sync_script="./sync_output_dir.sh"
  if [[ -f "$sync_script" ]]; then
    echo "Starting advanced background sync using $sync_script..."
    # 使用 stdbuf 确保日志实时输出，同步间隔设置为 600s (可按需调整)
    stdbuf -oL -eL bash "$sync_script" "$TEMP_OUTPUT_DIR" "$FINAL_OUTPUT_DIR" 600 &
    SYNC_PID=$!
    echo "Background sync started with PID: $SYNC_PID"
    trap "kill $SYNC_PID; echo 'Sync process $SYNC_PID terminated.'" EXIT
  else
    echo "Warning: $sync_script not found! Falling back to simple sync loop."
    while true; do
      rsync -avz --delete "$TEMP_OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"
      sleep 60
    done &
    SYNC_PID=$!
    trap "kill $SYNC_PID; echo 'Simple sync process terminated.'" EXIT
  fi
}

# 解析命令行参数
parse_args "$@"
# 准备模型和目录
prepare_inputs

LLAMA_FACTORY_ARGS=(
  --deepspeed "../examples/deepspeed/ds_z2_config.json"
  --dataset "my_cpt_data"
  --model_name_or_path "$LOCAL_MODEL_PATH"
  --output_dir "$TEMP_OUTPUT_DIR"
  --max_steps "1000"
  --per_device_train_batch_size "1"
  --gradient_accumulation_steps "8"
  --trust_remote_code "true"
  --stage "pt"
  --do_train
  --finetuning_type "full"
  --dataset_dir "./test"
  --streaming "true"
  --cutoff_len "4096"
  --preprocessing_num_workers "64"
  --dataloader_num_workers "16"
  --logging_steps "10"
  --save_steps "1000"
  --plot_loss "true"
  --overwrite_output_dir "true"
  --save_only_model "false"
  --learning_rate "1.0e-4"
  --num_train_epochs "1.0"
  --lr_scheduler_type "cosine"
  --warmup_ratio "0.001"
  --bf16 "true"
  --ddp_timeout "180000000"
  --push_to_hub "false"
  --gradient_checkpointing "true"
  --report_to "wandb"
)

# 初始化环境并组装分布式参数
init_env

# 启动后台同步进程
start_sync_output

# 运行 torchrun 命令，直接展开使用前面准备好的 DIST_ARGS 数组
torchrun "${DIST_ARGS[@]}" \
  ../src/train.py \
  "${LLAMA_FACTORY_ARGS[@]}"

# 训练完成后，进行最后一次强制同步
echo "Training finished. Performing final sync..."
rsync -avz --delete "$TEMP_OUTPUT_DIR/" "$FINAL_OUTPUT_DIR/"

echo "Training script finished successfully."
