直接复制下面内容，粘贴到 GitHub 的 README 编辑器即可（或存成 README.md 放进仓库）：
markdown
# MobileViTv3-DSR

融合**差分注意力（Differential Attention, DA）**与**空间缩减注意力（Spatial Reduction Attention, SRA）**的改进 MobileViTv3 轻量级交通标志分类模型，面向复杂街景场景实现鲁棒识别。

> 对应论文：*Dual-Attention MobileViTv3: Differential and Spatial-Reduction Attention for Robust Traffic Sign Recognition in Complex Street Scenes*（面向复杂街景的城市交通标志鲁棒分类）

## 项目简介

城市交通标志的精准识别是智能交通系统与城市数字化治理的关键环节。现有轻量级模型在复杂街景下普遍存在**细粒度特征提取不足、背景干扰鲁棒性差、精度与效率难以兼顾**等问题。本项目在 MobileViTv3 基础上进行如下改进：

- 修正 MobileViTBlock 中卷积层的**通道维度匹配错误**，优化局部/全局特征流转逻辑；
- 将原始 SiLU 激活函数替换为 **GELU**，增强对交通标志细粒度特征的表征能力；
- 在 Transformer 模块前嵌入 **空间缩减注意力（SRA）**，对键值特征进行空间缩减，降低计算开销并过滤街景背景噪声；
- 用 **差分注意力（DA）** 替换标准多头自注意力，通过原始查询与偏移查询的差分运算，抑制背景响应、放大标志区域权重。

SRA 与 DA 串联形成 **“空间降维 → 背景抑制 → 标志聚焦”** 的递进增强，在轻量架构下兼顾精度与效率，可部署于车载终端与路侧边缘设备。

## 主要特点

- 轻量级架构，面向移动端与边缘设备实时推理
- 双注意力机制协同：SRA 前置降维降噪 + DA 前景/背景自适应区分
- 在 GTSRB 四分类子集上测试准确率达 **99.65%**，较原模型提升 **2.19 个百分点**
- 优于 GoogLeNet（+9.12pp）、MobileNetV2（+3.42pp）、Swin-Tiny（+6.23pp）等基线

## 方法概述

### 整体框架

Input ─▶ Stem ─▶ Stage1 ─▶ Stage2 ─▶ Stage3 ─▶ GAP ─▶ LayerNorm ─▶ Classifier (Softmax)
│ │ │
└── MobileViTv3 Block ──┘
├─ Local：DWConv-3×3 → Conv-1×1（纹理与形状特征）
└─ Global：SRA → DA-Transformer → Fold（全局建模）
plaintext

批归一化后的图像先经 Stem 卷积层提取边缘、形状、颜色等基础特征，再经三个阶段及步长为 2 的下采样提取多尺度特征；每个 MobileViTv3 Block 采用局部与全局联合建模，多尺度特征经全局平均池化、层归一化后送入分类头与 Softmax，输出 4 类交通标志概率。

### 空间缩减注意力（SRA）

对输入特征线性投影生成 Q/K/V 后，对键、值向量施加步长卷积实现**空间缩减**，降低特征序列长度与计算复杂度，同时过滤道路边缘、建筑物轮廓等背景噪声、强化标志特征响应，为后续差分注意力全局建模提供高效输入。

### 差分注意力（DA）

构造**原始查询**与**可学习偏移查询**两组分支，分别求解注意力分布后做**差分运算**，得到聚焦交通标志的差异化权重；数值越高代表对应位置交通标志特征越显著。与前端 SRA 串联协作，在不明显增加参数量的前提下提升细粒度类别识别鲁棒性。

## 数据集

使用 GTSRB（German Traffic Sign Recognition Benchmark）四分类子集，图像采集自德国城市真实街景行车记录仪视频帧，涵盖光照、天气、遮挡等复杂干扰条件。共 **5700 张**样本，按 **3:1:1** 划分训练集 / 验证集 / 测试集：

| 类别 | 训练集 | 验证集 | 测试集 | 总数 |
|---|---:|---:|---:|---:|
| 限速 | 900 | 300 | 300 | 1500 |
| 禁止超车 | 810 | 270 | 270 | 1350 |
| 让行 | 864 | 288 | 288 | 1440 |
| 主干道优先 | 846 | 282 | 282 | 1410 |
| **合计** | **3420** | **1140** | **1140** | **5700** |

## 实验结果

**消融实验（验证集准确率）：**

| 模型 | 验证集准确率 |
|---|---:|
| MobileViTv3 | 97.81% |
| MobileViTv3 + SRA | 98.16% |
| MobileViTv3 + DA | 96.32% |
| **MobileViTv3 + SRA + DA** | **99.82%** |

结果表明：单独嵌入 DA 或 SRA 均能提升性能，二者融合后验证准确率最高、收敛最快，对 No overtaking 与 Speed limit 等易混淆类别的区分能力显著优于对比模型，抗街景背景干扰能力更强。

## 环境依赖

| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11（64-bit） |
| 处理器 | Intel Core i9-14900 |
| 显卡 | NVIDIA GeForce RTX 5060 Laptop GPU |
| 内存 | 16 GB |
| 深度学习框架 | PyTorch |
| 编程语言 | Python 3.11 |
| 开发环境 | PyCharm 2023.3 |

## 快速开始

```bash
git clone https://github.com/Xhw07/MobileVitv3-DSR.git
cd MobileVitv3-DSR
pip install -r requirements.txt
训练与测试（命令请按实际代码结构调整）：
bash
# 训练
python train.py --data data/GTSRB --epochs 50

# 测试与指标评估
python test.py --checkpoint checkpoints/best.pt
目录结构（示意，请按实际代码调整）
plaintext
MobileVitv3-DSR/
├── data/                # GTSRB 四分类子集
├── models/              # MobileViTv3-DS 模型定义（含 SRA / DA 模块）
├── utils/               # 数据加载与工具函数
├── train.py             # 训练脚本
├── test.py              # 测试与评估脚本
├── requirements.txt     # 依赖清单
└── README.md
局限与展望
当前数据集仅覆盖四类常见交通标志，后续将扩展类别与复杂场景以提升泛化能力；
当前仅实现分类任务，后续将开发检测与分类联合算法，实现识别到定位预警升级；
将针对雨天反光、雾天模糊等极端天气导致的图像退化进行专项优化，并将模型嵌入智慧城市交通管理平台，构建识别、预警、处置闭环。
引用
如果本项目对你的研究有帮助，请引用：
bibtex
@misc{MobileVitv3-DSR,
  title  = {融合差分注意力与空间缩减注意力的MobileViTv3：面向复杂街景的城市交通标志鲁棒分类},
  author = {待补充},
  year   = {2026}
}
许可
本项目代码仅供学术研究使用（License 待定）。
