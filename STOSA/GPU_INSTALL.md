# GPU 环境安装指南

## 1. 使用清华镜像安装依赖

### 方式一：使用 requirements.txt 安装
```bash
# 进入项目目录
cd /path/to/STOSA

# 使用清华镜像安装所有依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方式二：单独安装 PyTorch（推荐，支持 GPU）
根据你的 CUDA 版本选择合适的 PyTorch 版本：

#### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### CUDA 12.4（最新）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### CPU 版本（如果没有 GPU）
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方式三：完整安装脚本
```bash
# 1. 安装其他依赖
pip install numpy scipy tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 安装 PyTorch（根据你的 CUDA 版本选择上面的命令）
# 例如：CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 验证安装

### 检查 PyTorch 和 CUDA 是否可用
```python
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
```

### 检查其他依赖
```python
import numpy
import scipy
import tqdm

print(f"NumPy 版本: {numpy.__version__}")
print(f"SciPy 版本: {scipy.__version__}")
print(f"tqdm 版本: {tqdm.__version__}")
```

## 3. 运行训练

### 在 GPU 上运行训练
```bash
cd /path/to/STOSA

# 使用 GPU 训练（默认会自动使用可用的 GPU）
python main.py \
    --model_name=DistSAModel \
    --data_name=Beauty \
    --output_dir=outputs/ \
    --lr=0.001 \
    --hidden_size=64 \
    --max_seq_length=100 \
    --hidden_dropout_prob=0.3 \
    --num_hidden_layers=1 \
    --weight_decay=0.0 \
    --num_attention_heads=4 \
    --attention_probs_dropout_prob=0.0 \
    --pvn_weight=0.005 \
    --epochs=500 \
    --gpu_id=0
```

### 指定 GPU
如果你有多张 GPU，可以通过 `--gpu_id` 参数指定：
```bash
# 使用第 1 张 GPU
python main.py ... --gpu_id=0

# 使用第 2 张 GPU
python main.py ... --gpu_id=1
```

### 强制使用 CPU（即使有 GPU）
```bash
python main.py ... --no_cuda
```

## 4. 其他常用镜像源

### 阿里云镜像
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 中科大镜像
```bash
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

### 豆瓣镜像
```bash
pip install -r requirements.txt -i https://pypi.douban.com/simple/
```

## 5. Conda 环境（可选）

如果你使用 Conda，可以这样创建环境：

```bash
# 创建 conda 环境
conda create -n stosa python=3.9 -y
conda activate stosa

# 安装 PyTorch（CUDA 12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install numpy scipy tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 6. 常见问题

### Q: 提示 CUDA out of memory 怎么办？
A: 减小 batch_size，例如从 256 改为 128 或 64：
```bash
python main.py ... --batch_size=128
```

### Q: 如何查看 GPU 使用情况？
A: 使用 nvidia-smi 命令：
```bash
nvidia-smi
```

### Q: 训练速度很慢怎么办？
A: 
1. 确保使用了 GPU（检查 torch.cuda.is_available()）
2. 增加 num_workers（如果数据加载慢）
3. 使用更快的 GPU