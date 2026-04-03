#!/bin/bash
set -x # 打印执行的命令，方便调试
# Track background helper processes for cleanup
# AUTO_EVAL_PID=""
# SYNC_OUTPUT_PID=""

init_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY # 清除代理环境变量
  # 安装或升级各种 Python 库
  #pip install "deepspeed>=0.10.0,<=0.16.9"
  # --- 在这里添加你的 LLaMA-Factory 安装和 metrics 依赖 ---
  # 获取脚本所在的目录
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  # LLaMA-Factory 的根目录是脚本目录的上一级
  LLAMA_FACTORY_ROOT="$(dirname "$SCRIPT_DIR")" 
  echo "Changing directory to LLaMA-Factory root: $LLAMA_FACTORY_ROOT for pip installs."
  cd "$LLAMA_FACTORY_ROOT"
  echo "Installing LLaMA-Factory in editable mode..."
  #pip install -e . || true
  echo "Installing dependencies from requirements/metrics.txt..."
  #pip install -r requirements/metrics.txt || true # 安装 requirements.txt 中的依赖，失败不中断
  # 安装完成后，切换回脚本所在的目录
  echo "Changing back to script directory: $SCRIPT_DIR."
  cd "$SCRIPT_DIR"
  # ------------------------------------------------------------------

  nvidia-smi # 打印 GPU 信息，确认 GPU 可用
  #llamafactory-cli version #检查llamafactory是否安装成功


  # 获取分布式训练相关的环境变量，通常来自集群调度系统（如 Arnold）
  WORKER_NUM=${ARNOLD_WORKER_NUM:-1} # 总节点数，默认为 1
  WORKER_GPU=${ARNOLD_WORKER_GPU:-1} # 每节点 GPU 数，默认为 1
  WORKER0_HOST=${ARNOLD_WORKER_0_HOST:-'127.0.0.1'} # 主节点 IP，默认为本地
  #WORKER0_PORT=${ARNOLD_WORKER_0_PORT:-'29500'} # 主节点端口，默认为 29500
  WORKER0_PORT=${ARNOLD_WORKER_0_PORT:-$(find_free_port)}
  if [ "$local_run" = true ]; then # 如果是本地运行模式，强制主节点 IP 为 127.0.0.1
    WORKER0_HOST='127.0.0.1'
  fi
  NPROC_PER_NODE="$WORKER_GPU" # 每节点进程数（即 GPU 数）
  MASTER_ADDR="$WORKER0_HOST" # 主节点 IP
  MASTER_PORT="$(cut -d',' -f1 <<<"$WORKER0_PORT")" # 从 WORKER0_PORT 中提取第一个端口
  if [[ -z "$MASTER_PORT" || "$MASTER_PORT" == "$WORKER0_PORT" ]]; then MASTER_PORT="$WORKER0_PORT"; fi # 确保 MASTER_PORT 有值
  
  # 组装 torchrun 的分布式参数
  DIST_ARGS=(
    --nnodes="$WORKER_NUM"
    --node_rank="${ARNOLD_ID:-0}" # 当前节点在集群中的排名，默认为 0
    --nproc_per_node="$NPROC_PER_NODE"
    --master_addr="$MASTER_ADDR"
    --master_port="$MASTER_PORT"
  )
}
# torchrun --master_port 29500 --nproc_per_node=8 --nnodes=2 --node_rank=0  --master_addr=192.168.0.1  train.py
# torchrun --master_port 29500 --nproc_per_node=8 --nnodes=2 --node_rank=1  --master_addr=192.168.0.1  train.py

# 解析命令行参数
  
parse_args() {
  # 1. 定义所有参数的默认值
  # 没想好怎么处理dataset_path
  dataset_path='/mnt/hdfs/data/'
  model_name_or_path='hdfs://harunava/nlp/yujunshuai.08/local_service/llm/base_models/Qwen3-0.6B' # 假设默认 HDFS 输出路径，实际应该映射到dataset目录下的模型
  output_dir='./output' # 这个路径会挂载到一个hdfs下，/mnt/hdfs/local_service_llm/arnold_output
  max_steps='1000'
  per_device_train_batch_size='1'
  gradient_accumulation_steps='8'
  deepspeed_config='examples/deepspeed/ds_z2_config.json'
  wandb_name='ttls_poi_cpt'
  # 内部变量，用于存储本地模型和数据集路径
  _local_model_path=""
  _local_dataset_name="my_cpt_data"
  # 初始化 extra_llamafactory_args 变量，用于收集其他参数
  extra_llamafactory_args=""

  # 2. 解析命令行参数，覆盖默认值
  # 如果没有传入任何参数，则显示使用说明，我没定义usage，注释掉
  # if [ $# -eq 0 ]; then
  #   usage
  # fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset_path) dataset_path="$2"; echo "dataset_path: ${dataset_path}"; shift; shift ;;
      --model_name_or_path) model_name_or_path="$2"; echo "model_name_or_path: ${model_name_or_path}"; shift; shift ;;
      --output_dir) output_dir="$2"; echo "output_dir: ${output_dir}"; shift; shift ;;
      --max_steps) max_steps="$2"; echo "max_steps: ${max_steps}"; shift; shift ;;
      --per_device_train_batch_size) per_device_train_batch_size="$2"; echo "per_device_train_batch_size: ${per_device_train_batch_size}"; shift; shift ;;
      --gradient_accumulation_steps) gradient_accumulation_steps="$2"; echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"; shift; shift ;;
      --deepspeed_config) deepspeed_config="$2"; echo "deepspeed_config: ${deepspeed_config}"; shift; shift ;;
      --wandb_project) wandb_project="$2"; echo "wandb_project: ${wandb_project}"; shift; shift ;;
      --wandb_name) wandb_name="$2"; echo "wandb_name: ${wandb_name}"; shift; shift ;;
      *) # 捕获所有未明确解析的参数，并将其添加到 extra_llamafactory_args
      if [[ $# -ge 2 && "$2" != --* ]]; then
          extra_llamafactory_args+=" $1 $2"
          shift; shift
        else
          extra_llamafactory_args+=" $1"
          shift
        fi
        ;;
      esac
    done

  # 3. 移除强制参数检查，因为现在所有参数都有默认值


  #train_args_overrides="${train_args_overrides} ${extra_args_cli}" # 合并额外的 CLI 参数，没懂为什么要，先删

  # echo "train: ${train}"
  # echo "auto_eval: ${auto_eval}"

  # 设置 WandB 相关的环境变量
  export WANDB_PROJECT="${wandb_project}"
  rundt="$(date "+%Y_%m%d_%H%M")"
  if [ ${ARNOLD_TRIAL_ID:-1} == 1 ]; then
    wandb_name="${wandb_name}_test_${rundt}"
  else
    wandb_name="${wandb_name}_${ARNOLD_TRIAL_ID}"
  fi
  export WANDB_NAME="${wandb_name}"
  export HDFS_DATASET_PATH="${hdfs_dataset_path}" #这是什么

  # 设置输出目录
  if [[ -n "$output_dir" ]]; then
    output_dir="${output_dir}/${wandb_name}"
  else
    output_dir="/mnt/hdfs/local_service_llm/arnold_output/${wandb_name}"
  fi
  mkdir -p "$output_dir"

  # 设置临时输出目录，如果直接写那个挂在目录，会卡死，还是相当于网络传输hdfs
  if [[ -z "$temp_output_dir" ]]; then
    temp_output_dir="$(pwd)/saved_models"
  fi
  mkdir -p "$temp_output_dir"
}

# 准备输入数据和模型
prepare_inputs() {
  # 获取脚本所在的目录
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  # LLaMA-Factory 的根目录是脚本目录的上一级
  LLAMA_FACTORY_ROOT="$(dirname "$SCRIPT_DIR")" 

  # --- 处理 model_name_or_path ---
  local_models_root="${LLAMA_FACTORY_ROOT}/model" # 本地模型存储根目录
  mkdir -p "$local_models_root"

  if [[ "$model_name_or_path" == hdfs://* ]]; then # 如果是 HDFS 路径
    local_model_name="$(basename "$model_name_or_path")"
    _local_model_path="${local_models_root}/${local_model_name}"
    echo "Downloading model from HDFS: $model_name_or_path to $_local_model_path"
    if [[ -e "$_local_model_path" ]]; then rm -rf "$_local_model_path"; fi # 如果本地已存在，则删除
    hdfs dfs -copyToLocal "$model_name_or_path" "$local_models_root" || hdfs dfs -get "$model_name_or_path" "$local_models_root"
  elif [[ -d "$model_name_or_path" ]]; then # 如果是本地目录
    _local_model_path="$model_name_or_path"
  else
    echo "Error: Model path '$model_name_or_path' is neither a valid HDFS path nor a local directory."
    exit 1
  fi

  # --- 处理 dataset_path --- 先不做，弄不清楚
}

start_sync_output() {
  sync_script="sync_output_dir.sh"
  # Run in background with line-buffered stdout/stderr; logs are prefixed internally by the script
  stdbuf -oL -eL bash "$sync_script" "$temp_output_dir" "$output_dir" 600 &
  SYNC_OUTPUT_PID="$!"
  echo "log of run_sft_train: started sync_output_dir pid=${SYNC_OUTPUT_PID}"
}

run_train() {
  # 设置环境变量
  if [ "$local_run" = true ]; then
    export NCCL_NET_PLUGIN=none
    export NCCL_NET=Socket
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
  fi

  # 定义参数数组
  cmd_args=(
      src/train.py
      # 核心参数，使用 prepare_inputs 处理后的本地路径和 parse_args 中的值
      --deepspeed "${deepspeed_config}"                
      --dataset "${_local_dataset_name}" # 使用本地数据集路径
      --model_name_or_path "${_local_model_path}" # 使用本地模型路径
      --output_dir "${temp_output_dir}" # 训练结果输出到本地临时目录
      --max_steps "${max_steps}"
      --per_device_train_batch_size "${per_device_train_batch_size}"
      --gradient_accumulation_steps "${gradient_accumulation_steps}"
      # LLaMA-Factory 训练脚本的其他固定参数或你希望默认开启的参数
      # model
      --trust_remote_code "true"  
      # method
      --stage pt
      --do_train
      --finetuning_type "full"
      # dataset
      --dataset_dir "./ttls_poi_cpt/"
      --streaming "true" 
      --cutoff_len "4096"
      --preprocessing_num_workers "64"
      --dataloader_num_workers "16"
      # output
      --logging_steps "10"
      --save_steps "1000"
      --plot_loss "true"
      --overwrite_output_dir "true"
      --save_only_model "false"
      --report_to "wandb"
      --run_name "${wandb_name}"  
      # train
      --learning_rate "1.0e-4"
      --num_train_epochs "1.0"
      --lr_scheduler_type "cosine"
      --warmup_ratio "0.001"
      --bf16 "true"
      --ddp_timeout "180000000"
      # others
      --push_to_hub "false"
      --gradient_checkpointing "true"
      --seed "42"
      # 将从命令行收集到的额外参数添加到这里
      "${extra_llamafactory_args}"
  )

  # 执行命令
  if [[ "$use_mca" == "1" || "$use_mca" == "true" ]]; then
    USE_MCA=1 torchrun "${DIST_ARGS[@]}" "${cmd_args[@]}"
  else
    torchrun "${DIST_ARGS[@]}" "${cmd_args[@]}"
  fi
}


# Graceful cleanup on exit/signals
cleanup() {
  echo "log of run_sft_train: cleanup starting..."
  
  # 检查并杀掉 sync_output 进程
  if [[ -n "${SYNC_OUTPUT_PID:-}" ]]; then
    if kill -0 "$SYNC_OUTPUT_PID" 2>/dev/null; then
      echo "Killing sync_output process (PID: $SYNC_OUTPUT_PID)"
      kill "$SYNC_OUTPUT_PID" 2>/dev/null || true
    else
      echo "sync_output process (PID: $SYNC_OUTPUT_PID) already dead."
    fi
  fi
  
  echo "log of run_sft_train: cleanup done"
}

# 查找可用端口
find_free_port() {
  python -c 'import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

# --- 主函数 ---
main() {
  parse_args "$@"
  init_env
  prepare_inputs
  # 注册信号捕获，确保脚本退出（无论是正常结束、被 Kill、还是报错退出）都会运行 cleanup
  trap 'cleanup' EXIT INT TERM
  # if [ "${METIS_TASK_INDEX}" = "0" ]; then
  #   start_sync_output
  # fi
  # 运行训练
  run_train
  # 【关键修改】获取 run_train 的退出码
  TRAIN_EXIT_CODE=$?
  if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT_CODE. Exiting immediately."
    exit $TRAIN_EXIT_CODE
    # 此时会触发 trap cleanup
  fi
  echo "Training finished successfully."
}

main "$@"
# 只有训练成功才会执行到这里
echo "Script finished, sleeping for 30 mins to keep container alive..."
sleep 1800
# sleep 结束后，脚本彻底退出，再次触发 trap cleanup（虽然进程可能已经被杀过一次，但 cleanup 里有检查，很安全）
