# AI Forgery Detection

统一多任务 AI 伪造图像分析项目。

当前仓库面向**团队内部协作与课程验收**，目标是在同一套模型中完成三类能力：

- **真假分类**：判断图像是真实还是 AI 生成 / 伪造
- **伪造解释**：给出伪造原因或异常线索描述
- **痕迹定位**：输出可疑区域 mask

最终汇总指标：

- 分类：`Acc`、`AP`
- 解释：`BLEU`、`ROUGE-L`
- 定位：`IoU`、`PixP`、`PixR`、`PixF1`

## 适用范围

这是一个**团队内部项目仓库**，README 重点服务于以下场景：

- 新成员快速完成环境准备与首轮跑通
- 团队成员统一训练、评估、推理命令
- 提供课程验收相关的数据组织与运行约定
- 为后续汇报、答辩和协作开发提供入口说明

不以对外开源发布为目标，因此这里默认你已经具备：

- 仓库访问权限
- 内部数据目录访问权限
- 基本的 Python / PyTorch 使用经验

---

## 项目概览

当前实现是一个**统一多任务模型**，不是三个彼此独立的模型：

- 共享骨干：`ResNet50`
- 分类头：输出真假分类 logits
- 定位头：输出可疑区域 mask logits
- 解释头：输出 explanation feature，再通过候选文本库进行检索式解释

训练采用两阶段策略：

- **Stage A**：先在多来源分类数据上做真假分类热身
- **Stage B**：再联合分类、解释、定位三类监督继续训练

项目同时支持 **partial labels**：

- 分类数据只参与分类损失
- SynthScars 数据同时参与分类 / 解释 / 定位损失
- 课程验收数据仅用于最终评估，不参与训练

---

## 仓库结构

```text
ai-forgery-detection/
├─ configs/
│  ├─ multitask.yaml
│  └─ multitask_smoke.yaml
├─ data/
│  ├─ imagenet_ai_0508_adm/
│  ├─ imagenet_ai_0419_biggan/
│  ├─ SynthScars/
│  └─ 2026-课设数据集/
├─ docs/
│  ├─ 项目过程记录.md
│  └─ 训练推理架构图.html
├─ src/
├─ train_multitask.py
├─ evaluate_multitask.py
├─ infer_multitask.py
└─ requirements.txt
```

核心脚本：

- `train_multitask.py`：训练入口
- `evaluate_multitask.py`：评估入口
- `infer_multitask.py`：单图推理入口
- `configs/multitask.yaml`：正式训练配置
- `configs/multitask_smoke.yaml`：最小可跑通配置

补充文档：

- `docs/项目过程记录.md`：过程记录、阶段结论、实验观察
- `docs/训练推理架构图.html`：训练 / 推理 / 线上增强思路的可视化说明

---

## 数据约定

### 1. 分类训练数据

`configs/multitask.yaml` 当前配置了两个分类数据源：

- `data/imagenet_ai_0508_adm`
- `data/imagenet_ai_0419_biggan`

脚本会从每个 root 下读取如下结构：

```text
<root>/
├─ train/
│  ├─ ai/
│  └─ nature/
└─ val/
   ├─ ai/
   └─ nature/
```

其中：

- `ai` → 标签 1
- `nature` → 标签 0

### 2. SynthScars 训练数据

用于 explanation + localization，同时也参与 fake 分类监督。

期望结构：

```text
SynthScars/
└─ train/
   ├─ images/
   └─ annotations/
      └─ train.json
```

以及测试集：

```text
SynthScars/
└─ test/
   ├─ images/
   └─ annotations/
      └─ test.json
```

### 3. 课程验收数据

`data/2026-课设数据集/` 默认作为**最终验收评估集**，按三个独立任务组织：

- `任务一：检测`
- `任务二：解释`
- `任务三：痕迹定位`

注意：课程验收数据不是“同一批图同时具备三种标注”的统一评估集，而是三个独立子集，因此评估时会分别跑三部分再汇总指标。

---

## 环境准备

### 1. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 提示执行策略限制，可先执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 2. 安装依赖

`requirements.txt` 只包含通用依赖，不包含 `torch` / `torchvision`。

先安装合适的 PyTorch，再安装其余依赖：

```powershell
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install -r requirements.txt
```

当前 `requirements.txt` 仅包含：

- `pillow`
- `pyyaml`
- `tqdm`

如果你使用的 CUDA 版本不是 `cu128`，请自行替换为本机对应版本。

### 3. 检查 CUDA 是否可用

```powershell
python -c "import torch; print('torch_version =', torch.__version__); print('cuda_available =', torch.cuda.is_available()); print('cuda_version =', torch.version.cuda); print('device_count =', torch.cuda.device_count()); print('device_name =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

如果训练机配置正常，至少应看到：

- `cuda_available = True`
- `device_count >= 1`

---

## 快速开始

推荐第一次拉仓库后按下面顺序执行。

### 1. 跑 smoke test

先验证：

- 数据能读取
- 模型能前向传播
- checkpoint 能保存
- 指标计算流程能跑通

```powershell
python train_multitask.py --config configs/multitask_smoke.yaml --output outputs_smoke
```

输出目录：

- `outputs_smoke/latest.pt`
- `outputs_smoke/best.pt`

### 2. 正式训练

```powershell
python train_multitask.py --config configs/multitask.yaml --output outputs
```

输出目录：

- `outputs/latest.pt`
- `outputs/best.pt`

### 3. 课程验收评估

如果目标是课程最终结果，建议显式指定 `course`：

```powershell
python evaluate_multitask.py --config configs/multitask.yaml --checkpoint outputs/best.pt --target course
```

说明：

- `evaluate_multitask.py` 的默认 `--target` 是 `synthscars_test`
- 如果你不加 `--target course`，默认评估的是 SynthScars test，而不是课程验收集

### 4. SynthScars 测试集评估

```powershell
python evaluate_multitask.py --config configs/multitask.yaml --checkpoint outputs/best.pt --target synthscars_test
```

### 5. 单图推理

```powershell
python infer_multitask.py path/to/image.png --config configs/multitask.yaml --checkpoint outputs/best.pt
```

推理输出包括：

- `label`
- `fake_score`
- `explanation`
- `mask_path`
- `explanation_source`

同时会在输入图片同目录下生成：

- `<image_stem>_pred_mask.png`

### 6. 远程 explanation 增强（可选）

如果你希望在本地模型输出之后，再用阿里云 `qwen3-vl-plus` 对 explanation 做增强，推荐在**项目内本地配置文件**里放 key，而不是改公共配置。

1. 复制一份本地覆盖配置：

```powershell
Copy-Item configs/multitask.local.example.yaml configs/multitask.local.yaml
```

2. 编辑 `configs/multitask.local.yaml`，填入你的 key：

```yaml
inference:
  remote_explanation:
    enabled: true
    api_key: "你的阿里云 Model Studio Key"
```

说明：

- `load_config()` 会先读取 `configs/multitask.yaml`
- 如果存在 `configs/multitask.local.yaml`，会自动把它覆盖合并进来
- `configs/*.local.yaml` 已加入 `.gitignore`，不会提交到仓库

3. 然后直接运行：

```powershell
python infer_multitask.py path/to/image.png --config configs/multitask.yaml --checkpoint outputs/best.pt
```

如果你更喜欢环境变量，也仍然支持：

```powershell
$env:DASHSCOPE_API_KEY="你的阿里云 Model Studio Key"
```

此时：

- `label`、`fake_score`、`mask_path` 仍来自本地模型
- `explanation` 会优先使用 `qwen3-vl-plus` 生成的增强结果
- `explanation_source` 会标记为 `remote`、`local` 或 `local_fallback`
- 如果既没有配置 `remote_explanation.api_key`，也没有配置 `DASHSCOPE_API_KEY`，脚本会自动回退到本地 explanation

---

## 配置说明

当前正式配置文件为 `configs/multitask.yaml`，其中关键项包括：

### model

- `image_size_cls`
- `image_size_seg`
- `backbone_out_channels`
- `segmentation_hidden_channels`
- `explanation_feature_dim`

### train

- `seed`
- `batch_size`
- `num_workers`
- `mixed_precision`
- `learning_rate`
- `weight_decay`
- `epochs_stage_a`
- `epochs_stage_b`
- `cls_weight`
- `loc_weight`
- `exp_weight`
- `imagenet_fake_sample_limit`
- `imagenet_real_sample_limit`
- `synthscars_val_ratio`

### data

- `imagenet_roots`
- `synthscars_root`
- `course_root`

### inference

- `remote_explanation.enabled`
- `remote_explanation.provider`
- `remote_explanation.model`
- `remote_explanation.api_base`
- `remote_explanation.api_key`
- `remote_explanation.api_key_env`
- `remote_explanation.timeout_sec`
- `remote_explanation.fallback_to_local`

如果团队成员更换机器或目录布局，通常只需要先检查和修改 `data` 部分路径。若要启用线上 explanation，推荐把私有 key 放在 `configs/multitask.local.yaml` 中，只把公共开关和公共接口地址保留在 `configs/multitask.yaml`。

---

## 运行行为说明

### 1. 训练默认要求 CUDA

`train_multitask.py` 和 `evaluate_multitask.py` 默认都会要求 CUDA 可用。

也就是说：

- 如果当前环境没有识别到 GPU，会直接报错退出
- 不会静默回退到 CPU
- 只有显式加 `--allow-cpu` 时，才允许 CPU 运行

例如：

```powershell
python train_multitask.py --config configs/multitask_smoke.yaml --output outputs_smoke --allow-cpu
```

仅建议在调试或无 GPU 环境下临时使用。

### 2. 推理脚本会自动选择设备

`infer_multitask.py` 的行为与训练不同：

- 有 CUDA 时用 CUDA
- 没有 CUDA 时回退到 CPU

因此单图推理对环境要求更宽松。

如果同时启用了远程 explanation 增强，还会额外发起一次可选的线上多模态请求；这一步失败时会自动回退到本地 explanation，不影响本地分类和 mask 输出。

### 3. 训练输出

训练期间会打印：

- `A train x/y`
- `A val x/y`
- `B train x/y`
- `B val x/y`

并在每个 epoch 后输出汇总信息，包括：

- 当前 stage
- 当前 epoch
- train loss
- 当前 metrics
- 当前 batch 来源统计

同时持续更新：

- `latest.pt`
- `best.pt`

---

## 团队协作建议

### 1. 提交前建议至少做一项验证

按你本次修改范围，至少选择其一：

- 文档改动：检查 README 命令和路径是否仍可直接复制执行
- 训练逻辑改动：先跑 `multitask_smoke.yaml`
- 指标 / 评估改动：补跑一次对应 target 的评估命令
- 推理改动：至少跑一张图，确认输出结构没变

### 2. 不要把课程验收集混入训练

`data/2026-课设数据集/` 默认只用于最终评估。除非团队明确调整方案，否则不要把它接入训练流程。

### 3. 优先保持命令和 README 一致

如果修改了脚本参数、默认值、数据路径约定，请同步更新 README，避免内部协作时出现“文档能看但命令不能跑”的情况。

---

## 当前已知结论

结合当前记录，项目已有这些稳定结论：

- 统一多任务训练流程已经接线完成
- 修复后的定位监督相较早期版本有明显改进
- 新的正式训练结果在课程数据集上的分类与定位表现优于旧结果
- 当前分类瓶颈更像是**生成器覆盖不足**，而不是单纯的阈值问题
- explanation 目前本质上仍是检索式 baseline，稳定但语言表达上限有限

更详细的实验结论与过程说明见：

- `docs/项目过程记录.md`
- `docs/训练推理架构图.html`

---

## 常见问题

### 1. `ModuleNotFoundError: No module named 'tqdm'`

依赖未安装完整，重新执行：

```powershell
python -m pip install -r requirements.txt
```

### 2. `torch.cuda.is_available()` 是 `False`

通常是当前虚拟环境安装成了 CPU 版 PyTorch。可重新执行：

```powershell
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
```

### 3. `nvidia-smi` 正常，但 Python 仍识别不到 CUDA

优先检查当前虚拟环境中的 PyTorch 版本：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### 4. 训练太慢

建议先从 `multitask_smoke.yaml` 跑通，再逐步调大以下参数：

- `batch_size`
- `epochs_stage_a`
- `epochs_stage_b`
- `imagenet_fake_sample_limit`
- `imagenet_real_sample_limit`

---

## 推荐执行顺序

第一次接手这个仓库时，建议按以下顺序：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip uninstall -y torch torchvision
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m pip install -r requirements.txt
python train_multitask.py --config configs/multitask_smoke.yaml --output outputs_smoke
python train_multitask.py --config configs/multitask.yaml --output outputs
python evaluate_multitask.py --config configs/multitask.yaml --checkpoint outputs/best.pt --target course
```

如果只是验证脚本和环境是否正常，执行到 smoke test 即可。
