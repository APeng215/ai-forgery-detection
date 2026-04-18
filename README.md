# AI Forgery Detection

统一多任务 AI 伪造图像分析项目。当前目标是对**同一张图像**同时输出三项结果：
- **任务 1：真假分类**
- **任务 2：伪造解释**
- **任务 3：痕迹定位**

最终需要汇报的指标：
- 分类：`Acc`、`AP`
- 解释：`BLEU`、`ROUGE-L`
- 定位：`IoU`、`PixP`、`PixR`、`PixF1`

---

## 1. 项目结构

```text
ai-forgery-detection/
├─ configs/
│  ├─ multitask.yaml
│  └─ multitask_smoke.yaml
├─ data/
│  ├─ imagenet_ai_0508_adm/
│  ├─ SynthScars/
│  └─ 2026-课设数据集/
├─ docs/
├─ src/
├─ train_multitask.py
├─ evaluate_multitask.py
├─ infer_multitask.py
└─ requirements.txt
```

数据用途：
- `data/imagenet_ai_0508_adm/`、`data/imagenet_ai_0419_biggan/`：真假分类训练，可同时作为多个分类数据源接入
- `data/SynthScars/`：伪造解释 + 痕迹定位训练
- `data/2026-课设数据集/`：**最终课程验收，不参与训练**

---

## 2. 环境准备

### 2.1 创建虚拟环境

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 提示脚本权限问题，可以先执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

---

### 2.2 安装依赖

`requirements.txt` **只放通用依赖**，**不包含** `torch` 和 `torchvision`。  
这是为了避免把 GPU 环境误装成 CPU 版 PyTorch。

先安装 **CUDA 版 PyTorch**，再安装其余依赖：

```powershell
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install -r requirements.txt
```

当前项目在你的机器上已经验证过可识别：
- GPU：**NVIDIA GeForce RTX 5070 Ti**
- PyTorch CUDA：**cu128**

---

### 2.3 检查是否识别到显卡

```powershell
python -c "import torch; print('torch_version =', torch.__version__); print('cuda_available =', torch.cuda.is_available()); print('cuda_version =', torch.version.cuda); print('device_count =', torch.cuda.device_count()); print('device_name =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

如果输出里有：
- `cuda_available = True`
- `device_name = NVIDIA GeForce RTX 5070 Ti`

说明训练会走显卡。

---

## 3. 如何运行

### 3.1 先跑 smoke test

先跑一个最小训练，确认：
- 数据读取正常
- 多任务模型能前向传播
- checkpoint 能保存
- 指标计算流程能跑通

```powershell
python train_multitask.py --config configs/multitask_smoke.yaml --output outputs_smoke
```

输出目录：
- `outputs_smoke/latest.pt`
- `outputs_smoke/best.pt`

---

### 3.2 正式训练

```powershell
python train_multitask.py --config configs/multitask.yaml --output outputs
```

输出目录：
- `outputs/latest.pt`
- `outputs/best.pt`

---

### 3.3 评估

```powershell
python evaluate_multitask.py --config configs/multitask.yaml --checkpoint outputs/best.pt
```

评估目标指标：
- `Acc`
- `AP`
- `BLEU`
- `ROUGE-L`
- `IoU`
- `PixP`
- `PixR`
- `PixF1`

---

### 3.4 单图推理

```powershell
python infer_multitask.py path/to/image.png --config configs/multitask.yaml --checkpoint outputs/best.pt
```

推理会输出：
- 分类结果
- fake 分数
- explanation 文本
- 预测 mask 图片路径

---

## 4. 训练时会看到什么

训练脚本已经加了**明确的进度显示**：
- `A train x/y`：阶段 A 训练进度
- `A val x/y`：阶段 A 验证进度
- `B train x/y`：阶段 B 联合训练进度
- `B val x/y`：阶段 B 联合验证进度

进度条会实时显示：
- `loss`
- `cls_loss`
- `loc_loss`
- `exp_loss`

每个 epoch 结束后还会打印一行汇总，包括：
- 当前 stage
- 当前 epoch
- train loss
- 当前 metrics
- 当前 batch 来源统计

---

## 5. 显卡使用规则

`train_multitask.py` 默认会**强制要求 CUDA 可用**。

也就是说：
- 如果当前环境没有识别到显卡，训练会直接报错退出
- 不会悄悄回退到 CPU
- 只有你手动加 `--allow-cpu` 时，才允许 CPU 跑

例如：

```powershell
python train_multitask.py --config configs/multitask_smoke.yaml --output outputs_smoke --allow-cpu
```

正常训练时，**不要加 `--allow-cpu`**。

`evaluate_multitask.py` 也使用同样规则。

---

## 6. 当前训练设计

当前实现是一个**统一多任务模型**，不是三个独立模型。

- 共享骨干：`ResNet50`
- 分类头：真假分类
- 定位头：mask 预测
- 解释头：检索/模板式解释

训练策略：
- **阶段 A**：先用 `imagenet_ai_0508_adm` 做分类热身
- **阶段 B**：再和 `SynthScars` 做联合训练

这样做是为了让最终三项指标更稳，而不是只追求某一项指标。

---

## 7. 常见问题

### 7.1 报错 `ModuleNotFoundError: No module named 'tqdm'`
说明虚拟环境依赖没装完整，重新执行：

```powershell
python -m pip install -r requirements.txt
```

---

### 7.2 `torch.cuda.is_available()` 是 `False`
通常是因为装成了 CPU 版 torch。重新执行：

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

---

### 7.3 `nvidia-smi` 正常，但 Python 还是看不到 CUDA
这通常说明：
- 系统驱动没问题
- 但当前 `.venv` 里装的 PyTorch 版本不对

请优先检查：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

---

### 7.4 训练太慢
先从 `multitask_smoke.yaml` 开始，再逐步调大：
- `batch_size`
- `epochs_stage_a`
- `epochs_stage_b`
- `imagenet_fake_sample_limit`
- `imagenet_real_sample_limit`

---

## 8. 最推荐的执行顺序

第一次上手时，建议按这个顺序：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install -r requirements.txt
python train_multitask.py --config configs/multitask_smoke.yaml --output outputs_smoke
```

确认 smoke test 正常后，再跑正式训练：

```powershell
python train_multitask.py --config configs/multitask.yaml --output outputs
```
