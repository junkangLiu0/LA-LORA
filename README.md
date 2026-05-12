

# LA-LORA
# Rethinking LoRA for Privacy-Preserving Federated Learning in Large Models
恭喜《Rethinking LoRA for Privacy-Preserving Federated Learning in Large Models》被 ICLR 2026 接收！这个方向把 LoRA、大模型联邦学习和隐私保护结合起来，选题很有价值，也很符合当前大模型高效微调与隐私计算的发展趋势。

* 有代码问题+vx15653218567 马上回复！帮忙引用论文一下就行！

* 一张4090或者两张2080ti即可训练！！发顶会！！代码问题或者讨论+vx 15653218567

* 我的其他论文也都是这一套代码配置，均可复现！差分隐私，联邦泛化，联邦大模型，联邦优化，联邦大模型微调lora。。。。

* 个人主页：https://junkangliu0.github.io/



# LA-LORA：面向隐私保护联邦学习的 LoRA 微调代码

本仓库当前以 **`DP_lora.py`** 为主入口，支持在图像分类任务上运行 LoRA 联邦微调、差分隐私式梯度裁剪加噪，以及 LA-LORA 的本地交替更新机制。

> 本 README 根据当前 `DP_lora.py` 重新整理，只描述当前代码真实支持的内容；原 README 中关于 `main_lora.py`、`DP_LLM.py`、RoBERTa、GLUE/NLP 任务等内容不再写入，避免运行命令与代码不一致。

---

## 1. 项目功能概览

当前代码主要支持：

- **视觉任务**：`CIFAR10`、`CIFAR100`、`imagenet`（代码中对应 Tiny-ImageNet 风格数据）。
- **视觉骨干网络**：`swin_tiny`、`swin_small`、`swin_base`、`swin_large`、`VIT-B`、`VIT-L`，以及部分 ResNet 分支。
- **LoRA 微调**：对 Swin/ViT 的注意力模块添加 LoRA。
- **联邦学习**：使用 Ray 模拟多个客户端并行训练。
- **非 IID 数据划分**：通过 Dirichlet 分布模拟客户端数据异构。
- **隐私保护训练**：在本地梯度上进行裁剪和 Gaussian 噪声注入。
- **LA-LORA**：在客户端本地 step 内交替更新 `lora_B` 和 `lora_A`，并可选择对 LoRA 梯度使用 Gaussian low-pass filter。

---

## 2. 当前支持的算法

`DP_lora.py` 中主函数实际允许的 `--alg` 包括：

| 算法参数 | 代码含义 | 主要行为 |
|---|---|---|
| `DP-FedLORA` | DP-LoRA / FedLoRA 基线 | 同时更新 LoRA 参数，并进行 DP 式裁剪加噪。 |
| `DP-FedLORA-LS` | 带 Laplacian smoothing 的 LoRA 基线 | 在原有 LoRA 更新基础上使用旧版拉普拉斯平滑逻辑。 |
| `FFA-LORA` | Fixed-A / Freeze-A LoRA 基线 | 冻结 `lora_A`，主要更新 `lora_B` 和分类头。 |
| `AR-LORA` | 通信轮级别交替更新 LoRA | 按轮次交替更新 `lora_A` 与 `lora_B`，接近 RoLoRA 风格。 |
| `LA-LORA` | 本地 step 级别交替更新 LoRA | 在每个客户端本地 step 内交替更新 `lora_B` 和 `lora_A`，支持 `--la_filter`。 |
| `SAM-LORA` | SAM LoRA 基线 | 使用 SAM 优化器进行本地更新。 |

注意：虽然代码里的 `--alg` 默认值是 `FedAvg`，但主函数 `assert` 中并不允许直接运行 `FedAvg`。实际运行时请务必显式指定上表中的算法名。

---

## 3. LA-LORA 在当前代码中的实现

当前 `LA-LORA` 分支对应函数：

```python
update_FedAvg_AL(...)
```

其核心逻辑为：

```text
本地 step 1：更新 lora_B，冻结 lora_A
本地 step 2：更新 lora_A，冻结 lora_B
本地 step 3：更新 lora_B，冻结 lora_A
本地 step 4：更新 lora_A，冻结 lora_B
...
```

如果启用：

```bash
--la_filter 1
```

则代码会在 **DP 裁剪 + 加噪之后、optimizer.step() 之前**，对 LoRA 参数梯度使用固定 Gaussian low-pass filter：

```text
[1, 4, 6, 4, 1] / 16
```

其中：

- `lora_A`：沿输入特征维度做 1D 平滑；
- `lora_B`：沿输出维度做 1D 平滑；
- `--la_filter 0`：关闭该平滑，用于 LA-LORA(-filter) 消融实验。

---

## 4. 项目文件结构

建议目录结构如下：

```text
.
├── DP_lora.py                         # 主训练脚本
├── dirichlet_data.py                  # Dirichlet 非 IID 数据划分
├── dataset.py                         # Tiny-ImageNet 风格数据加载
├── model.py                           # Swin Transformer 模型定义
├── vit_model.py                       # ViT 模型定义
├── sam.py                             # SAM 优化器
├── optimizer/                         # 其他优化器实现
├── models/
│   ├── resnet.py
│   └── resnet_bn.py
├── data/                              # CIFAR 或 Tiny-ImageNet 数据
├── log/                               # 训练日志，自动创建
├── plot/                              # 训练曲线 npy，自动创建
├── model/                             # 最终模型参数，自动创建
└── checkpoint/                        # checkpoint 目录，自动创建
```

代码运行时会自动创建：

```text
./log
./plot
./model
./checkpoint
```

---

## 5. 环境配置

推荐使用 Conda：

```bash
conda create -n la_lora python=3.9 -y
conda activate la_lora
```

安装依赖：

```bash
pip install torch torchvision torchaudio
pip install ray opacus peft tensorboardX torchsummary matplotlib numpy filelock
```

如果项目中已有 `requirements.txt`，也可以直接：

```bash
pip install -r requirements.txt
```

---

## 6. 数据集说明

| 参数 | 数据集 | 加载方式 |
|---|---|---|
| `--data_name CIFAR10` | CIFAR-10 | 通过 `torchvision.datasets.CIFAR10` 自动下载到 `./data`。 |
| `--data_name CIFAR100` | CIFAR-100 | 通过 `torchvision.datasets.CIFAR100` 自动下载到 `./data`。 |
| `--data_name imagenet` | Tiny-ImageNet 风格数据 | 通过 `dataset.py` 中的 `TinyImageNetDataset` 读取。 |

CIFAR 数据无需手动准备，代码会自动下载。

如果运行 Tiny-ImageNet 风格数据，请放置为：

```text
./data/tiny-imagenet-200/
```

或者通过：

```bash
--datapath /your/data/root
```

指定数据根目录。

---

## 7. 模型说明

当前 LoRA 配置只针对 Swin/ViT 模型：

```python
target_modules=['attn.qkv', 'attn.proj']
```

推荐使用：

| 参数 | 说明 |
|---|---|
| `--CNN swin_tiny` | 速度较快，适合调试和 CIFAR-100 实验。 |
| `--CNN swin_base` | 模型更大，更接近论文中的大视觉模型设置。 |
| `--CNN VIT-B` | ViT-Base 实验。 |
| `--CNN VIT-L` | ViT-Large 实验。 |

代码中也存在 ResNet 分支，但当前 LoRA 配置没有为 ResNet 设置 target modules。如果要跑 ResNet，请使用：

```bash
--lora 0
```

否则可能出现 `lora_config` 未定义或 LoRA 无法正确注入的问题。

---

## 8. 预训练权重说明

如果 `--weights ""`，代码会跳过预训练权重加载。

如果需要加载预训练权重，当前代码虽然检查 `--weights` 是否存在，但实际 `torch.load()` 使用的是固定文件名。因此请确保对应权重文件在当前工作目录下。

| `--CNN` | 代码期望的权重文件名 |
|---|---|
| `swin_tiny` | `swin_tiny_patch4_window7_224.pth` |
| `swin_small` | `swin_small_patch4_window7_224.pth` |
| `swin_base` | `swin_base_patch4_window7_224_22k.pth` |
| `swin_large` | `swin_large_patch4_window7_224_22k.pth` |
| `VIT-B` | `vit_base_patch16_224_in21k.pth` |
| `VIT-L` | `jx_vit_large_patch16_224_in21k-606da67d.pth` |

如果只是想先跑通代码，建议直接使用：

```bash
--weights ""
```

---

## 9. 快速运行

### 9.1 最小测试命令

用于检查环境、Ray、数据加载和 LA-LORA 更新逻辑是否正常。

```bash
python DP_lora.py \
  --alg LA-LORA \
  --CNN swin_tiny \
  --data_name CIFAR100 \
  --num_workers 4 \
  --selection 0.5 \
  --p 1 \
  --num_gpus_per 0.25 \
  --batch_size 16 \
  --K 5 \
  --E 1 \
  --epoch 3 \
  --lr 0.1 \
  --lr_decay 0.99 \
  --alpha_value 0.1 \
  --r 16 \
  --lora 1 \
  --privacy 1 \
  --dp_sigma 0.2 \
  --C 0.2 \
  --la_filter 1 \
  --optimizer SGD \
  --gpu 0 \
  --weights ""
```

### 9.2 CIFAR-100 + Swin-Tiny + LA-LORA

```bash
python DP_lora.py \
  --alg LA-LORA \
  --CNN swin_tiny \
  --data_name CIFAR100 \
  --num_workers 8 \
  --selection 0.5 \
  --p 1 \
  --num_gpus_per 0.25 \
  --batch_size 16 \
  --K 20 \
  --E 1 \
  --epoch 100 \
  --lr 0.1 \
  --lr_decay 0.99 \
  --alpha_value 0.1 \
  --r 16 \
  --lora 1 \
  --privacy 1 \
  --dp_sigma 0.2 \
  --C 0.2 \
  --la_filter 1 \
  --optimizer SGD \
  --gpu 0 \
  --weights ""
```

### 9.3 CIFAR-100 + Swin-Base + LA-LORA

显存充足时可以运行：

```bash
python DP_lora.py \
  --alg LA-LORA \
  --CNN swin_base \
  --data_name CIFAR100 \
  --num_workers 8 \
  --selection 0.5 \
  --p 1 \
  --num_gpus_per 0.25 \
  --batch_size 16 \
  --K 20 \
  --E 1 \
  --epoch 100 \
  --lr 0.1 \
  --lr_decay 0.99 \
  --alpha_value 0.1 \
  --r 16 \
  --lora 1 \
  --privacy 1 \
  --dp_sigma 0.2 \
  --C 0.2 \
  --la_filter 1 \
  --optimizer SGD \
  --gpu 0 \
  --weights ""
```

如果你已经准备好 Swin-Base 预训练权重，可以把最后一行改为：

```bash
--weights ./swin_base_patch4_window7_224_22k.pth
```

---

## 10. 对比实验命令

为了公平对比，建议固定相同的模型、数据集、客户端数量、采样比例、学习率、LoRA rank 和 DP 参数，只替换 `--alg` 或 `--la_filter`。

### 10.1 DP-FedLORA / DP-LoRA 基线

```bash
python DP_lora.py \
  --alg DP-FedLORA \
  --CNN swin_tiny \
  --data_name CIFAR100 \
  --num_workers 8 \
  --selection 0.5 \
  --p 1 \
  --num_gpus_per 0.25 \
  --batch_size 16 \
  --K 20 \
  --E 1 \
  --epoch 100 \
  --lr 0.1 \
  --lr_decay 0.99 \
  --alpha_value 0.1 \
  --r 16 \
  --lora 1 \
  --privacy 1 \
  --dp_sigma 0.2 \
  --C 0.2 \
  --optimizer SGD \
  --gpu 0 \
  --weights ""
```

### 10.2 FFA-LORA 基线

```bash
python DP_lora.py \
  --alg FFA-LORA \
  --CNN swin_tiny \
  --data_name CIFAR100 \
  --num_workers 8 \
  --selection 0.5 \
  --p 1 \
  --num_gpus_per 0.25 \
  --batch_size 16 \
  --K 20 \
  --E 1 \
  --epoch 100 \
  --lr 0.1 \
  --lr_decay 0.99 \
  --alpha_value 0.1 \
  --r 16 \
  --lora 1 \
  --privacy 1 \
  --dp_sigma 0.2 \
  --C 0.2 \
  --optimizer SGD \
  --gpu 0 \
  --weights ""
```

### 10.3 AR-LORA / RoLoRA 风格基线

```bash
python DP_lora.py \
  --alg AR-LORA \
  --CNN swin_tiny \
  --data_name CIFAR100 \
  --num_workers 8 \
  --selection 0.5 \
  --p 1 \
  --num_gpus_per 0.25 \
  --batch_size 16 \
  --K 20 \
  --E 1 \
  --epoch 100 \
  --lr 0.1 \
  --lr_decay 0.99 \
  --alpha_value 0.1 \
  --r 16 \
  --lora 1 \
  --privacy 1 \
  --dp_sigma 0.2 \
  --C 0.2 \
  --optimizer SGD \
  --gpu 0 \
  --weights ""
```

### 10.4 LA-LORA(-filter) 消融实验

关闭 Gaussian low-pass filter，只保留本地交替更新：

```bash
python DP_lora.py \
  --alg LA-LORA \
  --la_filter 0 \
  --CNN swin_tiny \
  --data_name CIFAR100 \
  --num_workers 8 \
  --selection 0.5 \
  --p 1 \
  --num_gpus_per 0.25 \
  --batch_size 16 \
  --K 20 \
  --E 1 \
  --epoch 100 \
  --lr 0.1 \
  --lr_decay 0.99 \
  --alpha_value 0.1 \
  --r 16 \
  --lora 1 \
  --privacy 1 \
  --dp_sigma 0.2 \
  --C 0.2 \
  --optimizer SGD \
  --gpu 0 \
  --weights ""
```

### 10.5 LA-LORA 完整版本

启用本地交替更新和 Gaussian low-pass filter：

```bash
python DP_lora.py \
  --alg LA-LORA \
  --la_filter 1 \
  --CNN swin_tiny \
  --data_name CIFAR100 \
  --num_workers 8 \
  --selection 0.5 \
  --p 1 \
  --num_gpus_per 0.25 \
  --batch_size 16 \
  --K 20 \
  --E 1 \
  --epoch 100 \
  --lr 0.1 \
  --lr_decay 0.99 \
  --alpha_value 0.1 \
  --r 16 \
  --lora 1 \
  --privacy 1 \
  --dp_sigma 0.2 \
  --C 0.2 \
  --optimizer SGD \
  --gpu 0 \
  --weights ""
```

---

## 11. 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--alg` | `FedAvg` | 算法名。当前实际可用：`DP-FedLORA`、`DP-FedLORA-LS`、`FFA-LORA`、`LA-LORA`、`AR-LORA`、`SAM-LORA`。不要直接使用默认值。 |
| `--CNN` | `lenet5` | 模型名称。LoRA 推荐使用 `swin_tiny`、`swin_small`、`swin_base`、`swin_large`、`VIT-B`、`VIT-L`。 |
| `--data_name` | `CIFAR100` | 数据集名称：`CIFAR10`、`CIFAR100`、`imagenet`。 |
| `--num_workers` | `100` | 模拟客户端总数。 |
| `--selection` | `0.1` | 每轮参与训练的客户端比例。 |
| `--p` | `10` | 每轮客户端分组数量。简单运行建议设为 `1`。 |
| `--num_gpus_per` | `1` | 每个 Ray worker 占用的 GPU 比例，如 `0.25` 表示单卡最多 4 个 worker。 |
| `--batch_size` | `50` | 客户端本地 batch size。 |
| `--E` | `5` | 每个客户端本地 epoch 数。 |
| `--K` | `20` | 每个客户端每轮最多本地更新 step 数。 |
| `--epoch` | `1001` | 全局通信轮数。 |
| `--lr` | `0.1` | 客户端学习率。 |
| `--lr_decay` | `0.998` | 每轮学习率衰减系数。 |
| `--alpha_value` | `0.1` | Dirichlet 非 IID 参数。越小异构越强；等于 `1` 时走代码中的 IID 随机划分分支。 |
| `--lora` | `1` | 是否启用 LoRA。Swin/ViT 实验建议为 `1`。 |
| `--r` | `16` | LoRA rank，同时代码中 `lora_alpha=args.r`。 |
| `--freeze` | `0` | 是否冻结分类头。`0` 表示分类头可训练；`1` 表示更接近只训练 LoRA 参数。 |
| `--privacy` | `1` | 是否启用 DP 式裁剪加噪。 |
| `--C` | `0.2` | 初始裁剪阈值。训练中部分分支会将其动态改为 `min(median_grad_norm, 0.4)`。 |
| `--dp_sigma` | `0.2` | Gaussian 噪声系数。越大噪声越强，通常精度越低。 |
| `--la_filter` | `1` | 只对 `LA-LORA` 分支有效。`1` 启用 Gaussian low-pass filter；`0` 关闭。 |
| `--optimizer` | `SGD` | 优化器，可选 `SGD` 或 `AdamW`。 |
| `--gpu` | `0` | 使用的 GPU 编号，会写入 `CUDA_VISIBLE_DEVICES`。 |
| `--weights` | `./swin_tiny_patch4_window7_224.pth` | 预训练权重路径检查参数。设为 `""` 可跳过加载。 |
| `--preprint` | `10` | 每隔多少轮测试并写日志。 |
| `--datapath` | `./data` | 数据根目录，主要用于 Tiny-ImageNet 风格数据。 |

---

## 12. 输出文件

训练过程中会保存以下文件：

| 路径 | 内容 |
|---|---|
| `./log/*.txt` | 训练日志，包括测试精度、loss、学习率、DP 参数等。 |
| `./plot/*.npy` | 训练曲线数据，保存为 `(x, result, result_loss, test_list_loss)`。 |
| `./model/model_*` | 最终模型 `state_dict`。当前代码保存时默认没有 `.pth` 后缀。 |
| `runs/` | TensorBoard 日志目录，来自 `SummaryWriter(comment=alg)`。 |

查看 TensorBoard：

```bash
tensorboard --logdir runs
```

---

## 13. 差分隐私相关说明

当前代码中的隐私训练逻辑为：

```text
计算梯度 -> 梯度裁剪 -> 添加 Gaussian 噪声 -> 可选 Gaussian low-pass filter -> optimizer.step()
```

需要注意：

- `--privacy 1` 启用裁剪加噪。
- `--dp_sigma` 控制噪声强度。
- `--C` 是裁剪阈值，但代码中会在部分分支动态更新为 `min(median_grad_norm, 0.4)`。
- 代码虽然导入了 Opacus，但当前主训练路径主要使用手写的裁剪加噪逻辑。
- 当前代码没有自动把目标隐私预算 `epsilon=1/2/3` 反推出对应的 `dp_sigma`。

如果需要严格报告隐私预算，需要额外加入 RDP accountant 或单独标定 `--dp_sigma`。

---

## 14. 复现实验建议

建议至少运行以下方法，固定除算法外的所有参数：

| 实验目的 | 参数设置 |
|---|---|
| DP-LoRA 基线 | `--alg DP-FedLORA` |
| 固定 A 的基线 | `--alg FFA-LORA` |
| 轮级别交替基线 | `--alg AR-LORA` |
| LA-LORA 消融 | `--alg LA-LORA --la_filter 0` |
| LA-LORA 完整版本 | `--alg LA-LORA --la_filter 1` |

推荐统一基础参数：

```bash
--CNN swin_tiny \
--data_name CIFAR100 \
--num_workers 8 \
--selection 0.5 \
--p 1 \
--batch_size 16 \
--K 20 \
--E 1 \
--epoch 100 \
--lr 0.1 \
--lr_decay 0.99 \
--alpha_value 0.1 \
--r 16 \
--lora 1 \
--privacy 1 \
--dp_sigma 0.2 \
--C 0.2 \
--optimizer SGD \
--weights ""
```

---

## 15. 常见问题

### 15.1 报错：`weights file not exist`

说明当前指定的权重文件不存在。快速跑通时建议：

```bash
--weights ""
```

### 15.2 报错：`lora_config is not defined`

通常是因为 `--lora 1` 搭配了没有 LoRA 配置的模型。建议使用：

```bash
--CNN swin_tiny --lora 1
```

如果使用 ResNet，请设置：

```bash
--lora 0
```

### 15.3 显存不足

可以减少：

```bash
--batch_size 8
--selection 0.25
--num_gpus_per 0.5
--CNN swin_tiny
```

也可以先用 `--weights ""` 跳过预训练权重加载，检查流程是否能跑通。

### 15.4 默认参数不能直接跑

因为 `--alg` 默认是 `FedAvg`，但当前主函数只允许：

```text
DP-FedLORA, DP-FedLORA-LS, FFA-LORA, LA-LORA, AR-LORA, SAM-LORA
```

因此运行时必须显式指定 `--alg`。

---

## 16. 引用

如果使用该代码或方法，请引用对应论文：

```text
Rethinking LoRA for Privacy-Preserving Federated Learning in Large Models.
ICLR 2026.
```

---

## 17. License

请遵循原项目、依赖库以及预训练模型权重对应的许可证要求。


