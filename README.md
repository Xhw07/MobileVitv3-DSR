# MobileViTv3-DSR

融合**差分注意力（Differential Attention, DA）**与**空间缩减注意力（Spatial-Reduction Attention, SRA）**的轻量级交通标志分类模型，面向复杂街景场景。

> 对应论文：*Dual-Attention MobileViTv3: Differential and Spatial-Reduction Attention for Robust Traffic Sign Recognition in Complex Street Scenes*

## 结果概览

消融实验在 GTSRB 四分类子集的**验证集**上进行，模型量级是本仓库实测（输入 96×96）：

| 模型 | 验证集 Acc | 参数量 (M) | GFLOPs |
|---|---:|---:|---:|
| MobileViTv3（baseline） | 97.81% | 2.23 | 0.62 |
| + SRA | 98.16% | 3.36 | 0.67 |
| + DA | 96.32% | 2.48 | 0.76 |
| **+ SRA + DA（本文）** | **99.82%** | 3.61 | 0.81 |

测试集整体准确率 **99.65%**，较 GoogLeNet、MobileNetV2、Swin-Tiny 分别高 9.12、3.42、6.23 个百分点。

**复现记录**（本仓库当前实现，seed 42，batch 32，96×96，50 轮，未早停，RTX 5060 Laptop）：

| 划分 | 论文报告 | 本仓库复现 |
|---|---:|---:|
| 验证集（1140） | 99.82% | **99.91%** |
| 测试集（1140） | 99.65% | **99.82%** |

测试集 1140 张里只错 2 张（No overtaking → Yield 1 张、Priority road → Yield 1 张），Speed Limit 30 全对。逐类 P/R/F1/Specificity/AUC 与混淆矩阵见 [docs/results/dsr_metrics_test.json](docs/results/dsr_metrics_test.json)，逐轮 loss/acc 见 [docs/results/dsr_metrics.csv](docs/results/dsr_metrics.csv)。训练 50 轮约 18 min。

两点需要如实说明：单独加 SRA 有小幅提升，单独加 DA 反而掉到 96.32%——DA 的双分支注意力在缺少前置空间降维时不稳定，只有 SRA 先把键值序列压下来、抑制背景噪声后，DA 才能发挥前景/背景区分作用。这也是本模型采用 **SRA → DA 串联**而不是单独堆 DA 的原因。

## 方法

```text
Input → Stem(/2) → Stage1 → Stage2 → Stage3 → GAP → LayerNorm → FC → Softmax
                     │          │          │
                     └──────── MobileViTBlock ────────┘
                                  ├─ Local  : DWConv 3×3 → Conv 1×1
                                  └─ Global : Unfold → SRA → DA-Transformer → Fold
```

批归一化后的图像经 Stem 提取边缘/形状/颜色特征，再由三个阶段（步长 2 下采样）提取多尺度特征。每个 MobileViTBlock 同时建模局部与全局信息，最后经全局平均池化、层归一化与线性分类头输出 4 类概率。



**空间缩减注意力（SRA）**：查询取自完整 token 序列，键/值先经 `sr_ratio` 步长卷积做空间缩减，注意力在 `N × N'` 上计算。缩减后的尺寸直接取自卷积真实输出，不做整除假设；模块以残差方式注入 token 流。



**差分注意力（DA）**：

$$
A = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right) - \lambda \operatorname{softmax}\!\left(\frac{(Q + Q_{\text{off}})K^{\top}}{\sqrt{d}}\right)
$$

`Q_off` 由零初始化的线性层产生（可学习偏移查询，训练起点等价于恒等映射），`λ` 为每个注意力头独立可学习的标量。差分后的负响应被截断并重新归一化，保证聚合数值稳定。


**其他改动**：卷积与 MLP 中的 SiLU 统一替换为 GELU；全局分支沿用 MobileViT 的 unfold/fold，按 2×2 不重叠分块把 token 数降到 1/4，避免高分辨率阶段的注意力开销。

## 数据集

GTSRB 四分类子集（限速 / 禁止超车 / 让行 / 主干道优先），来自德国城市真实街景行车记录仪帧，含光照、天气、遮挡干扰。共 5700 张，按 3:1:1 划分：

| 类别 | 训练 | 验证 | 测试 | 合计 |
|---|---:|---:|---:|---:|
| Speed Limit 30 | 900 | 300 | 300 | 1500 |
| No overtaking (trucks) | 810 | 270 | 270 | 1350 |
| Yield | 864 | 288 | 288 | 1440 |
| Priority road | 846 | 282 | 282 | 1410 |
| **合计** | **3420** | **1140** | **1140** | **5700** |

原始 GTSRB 请从 [benchmark.ini.rub.de](https://benchmark.ini.rub.de/) 获取；本仓库不附带图像数据。

目录需整理成 `ImageFolder` 结构：

```text
GTSRB4/
├── train/
│   ├── Speed Limit 30/
│   ├── No overtaking (trucks)/
│   ├── Yield/
│   └── Priority road/
├── val/        # 同 train 的四个类别子目录
└── test/       # 同上
```

## 环境

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 (64-bit) |
| CPU | Intel Core i9-14900 |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB) |
| PyTorch / torchvision | 2.8.0+cu129 / 0.23.0 |
| Python | 3.11 |
| 训练开销 | batch 32、96×96 下约 21 s/epoch，50 轮约 18 min |

## 快速开始

```bash
git clone https://github.com/Xhw07/MobileVitv3-DSR.git
cd MobileVitv3-DSR
pip install -r requirements.txt

# 训练本文模型
python train.py --data-root /path/to/GTSRB4 --variant dsr --epochs 50

# 测试（自动读取 checkpoint 里记录的模型配置）
python test.py --data-root /path/to/GTSRB4 --checkpoint runs/dsr/best.pth --split test

# 参数量与计算量
python scripts/measure_complexity.py --img-size 96
```

复现消融表（四个变体：`baseline` / `sra` / `da` / `dsr`）。Windows 下一条命令跑完：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_all_variants.ps1 -DataRoot "D:/data/GTSRB4" -Epochs 50
```

也可以逐个手动跑：

```bash
python train.py --data-root /path/to/GTSRB4 --variant baseline --epochs 50
python test.py  --data-root /path/to/GTSRB4 --checkpoint runs/baseline/best.pth
```

常用参数：`--img-size`（默认 96）、`--batch-size`（32）、`--lr`（1e-4）、`--weight-decay`（2e-3）、`--label-smoothing`（0.1）、`--dropout`（0.1）、`--patch-size`（2）、`--sr-ratios`（`8,4,2`）、`--seed`（42）。验证与测试流程只做 resize + 归一化，不含随机增强，配合固定种子可复现同一结果。

训练产物写在 `--out-dir`（默认 `runs/<variant>`）：`best.pth`（含模型配置与类别名）、`metrics.csv`（逐轮 loss/acc）、`metrics_test.json`（逐类 P/R/F1/Specificity/AUC + 混淆矩阵）、`summary.json`。

## 目录结构

```text
MobileViTv3-DSR/
├── models/
│   ├── __init__.py
│   └── mobilevitv3_dsr.py    # SRA / DA / MobileViTBlock / 四种变体
├── utils/
│   ├── __init__.py
│   ├── data.py               # ImageFolder 与数据增强
│   ├── metrics.py            # 逐类 P/R/F1/Specificity/AUC（纯 NumPy 实现）
│   └── seed.py               # 固定种子与 DataLoader worker
├── scripts/
│   ├── measure_complexity.py # 参数量 / GFLOPs
│   └── run_all_variants.ps1  # 四变体消融一键脚本
├── docs/
│   ├── figures/              # 框架 / SRA / DA 示意图
│   └── results/              # 复现的指标 json 与逐轮 csv
├── train.py
├── test.py
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## 指标口径

`test.py` 输出逐类 Accuracy（one-vs-rest）、Precision、Recall、F1、Specificity 与 one-vs-rest ROC-AUC，并给出 macro 平均与整体准确率，同时保存混淆矩阵。指标全部用 NumPy 实现，不依赖 scikit-learn。

## 局限

- 仅覆盖四类常见交通标志，类别规模有限，复杂路口的多标志共存场景尚未验证；
- 仅做分类，未做端到端检测与定位；
- 未针对雨天反光、雾天模糊等极端退化做专项优化；
- 参数量较 baseline 增加约 1.4 M（+62%），FLOPs 增加约 0.19 G（+30%）。若部署端算力极紧，可把 SRA 的 `--sr-ratios` 调大或减小 `--patch-size` 退化路径，二者的取舍需在实际硬件上重新标定。

## 引用

```bibtex
@misc{mobilevitv3dsr2026,
  title  = {Dual-Attention MobileViTv3: Differential and Spatial-Reduction Attention for Robust Traffic Sign Recognition in Complex Street Scenes},
  author = {Xhw07},
  year   = {2026},
  url    = {https://github.com/Xhw07/MobileVitv3-DSR}
}
```

## 许可与致谢

代码基于 MIT 许可发布，见 [LICENSE](LICENSE)。

结构设计参考 Apple 的 MobileViT/MobileViTv3 与 PVT 的空间缩减注意力、以及 Differential Transformer 的差分注意力思路。本仓库只保留与本任务相关的实现，未包含上游仓库的检测/分割/蒸馏配置；如需完整上游代码请访问其官方仓库。
