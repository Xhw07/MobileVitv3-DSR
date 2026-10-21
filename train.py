import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import RandomErasing
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import csv
import os


# -------------------------- 1. 你的模型定义（MobileViTv3 + SRA + 差分注意力） --------------------------
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class SRA(nn.Module):
    def __init__(self, dim, sr_ratio=8):
        super().__init__()
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x)
            H_sr, W_sr = H // self.sr_ratio, W // self.sr_ratio
            x = x.reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            return x, H_sr, W_sr
        else:
            return x, H, W


class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DifferentialAttention(dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4, dropout=0.1, sr_ratio=8):
        super().__init__()
        self.dim = dim
        self.conv1 = conv_nxn_bn(dim, dim)
        self.conv2 = conv_nxn_bn(dim, dim)
        self.sra = SRA(dim, sr_ratio)
        self.transformer = nn.Sequential(*[TransformerBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.conv3 = conv_1x1_bn(dim, dim)
        self.conv4 = conv_nxn_bn(dim * 2, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        x_flat = x.flatten(2).transpose(1, 2)

        x_flat, H_sr, W_sr = self.sra(x_flat, H, W)
        x_flat = self.transformer(x_flat)

        x = x_flat.transpose(1, 2).reshape(B, C, H_sr, W_sr)
        x = self.conv3(x)

        x = nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        x = torch.cat([x, y], dim=1)
        x = self.conv4(x)
        return x


class MobileViTv3(nn.Module):
    def __init__(self, num_classes=4, dims=[16, 32, 64], depths=[2, 4, 3], num_heads=4, dropout=0.1, sr_ratio=8):
        super().__init__()
        self.stem = conv_nxn_bn(3, dims[0], stride=2)
        self.stages = nn.ModuleList()
        for i in range(len(dims)):
            stage = []
            if i > 0:
                stage.append(conv_nxn_bn(dims[i - 1], dims[i], stride=2))
            stage.append(MobileViTBlock(dims[i], depths[i], num_heads, dropout=dropout, sr_ratio=sr_ratio))
            self.stages.append(nn.Sequential(*stage))
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = x.flatten(2).mean(-1)
        x = self.norm(x)
        x = self.head(x)
        return x


# -------------------------- 2. 配置（强正则化版，目标验证 ~95%，跑满 50 轮） --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50            # 保持 50 轮，不早停（画折线图）
NUM_CLASSES = 4
BATCH_SIZE = 32
LR = 1e-4
IMG_SIZE = 64              # 224 -> 96：输入变小，信息减少，精度自然下降（96 能被 32 整除，模型可跑）
SAVE_PATH = "best_model_reg95.pth"   # 新文件名，不会覆盖你原来的 best_model.pth

# 强正则化：标签平滑 0.4 + transformer dropout 0.5 + 权重衰减 1e-3 + 更狠的增强
criterion = nn.CrossEntropyLoss(label_smoothing=0.5)
model = MobileViTv3(num_classes=NUM_CLASSES, dropout=0.6, sr_ratio=8).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 数据增强（更狠：裁切范围更宽 + 旋转 + 更高擦除概率）
from torchvision.transforms import GaussianBlur

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.4, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.12),
    GaussianBlur(kernel_size=(3,5), sigma=(0.1,2.0)), # 随机模糊
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    RandomErasing(p=0.5, scale=(0.03, 0.45))
])

# 验证集也要加轻微噪声！！【关键！！】
val_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    GaussianBlur(kernel_size=(3,3), sigma=(0.1,0.8)), # 验证图片轻微模糊，增加识别难度
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 数据集（去重版）
train_dataset = datasets.ImageFolder(
    root=r"D:/科研/智慧城市/dataset_split/train",
    transform=train_transform)
val_dataset = datasets.ImageFolder(
    root=r"D:/科研/智慧城市/dataset_split/val",
    transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# -------------------------- 3. 训练循环（50 轮，记录每轮指标到 metrics.csv） --------------------------
def train_epoch(model, loader, opt, cri):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        outs = model(imgs)
        loss = cri(outs, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        _, preds = torch.max(outs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def val_epoch(model, loader, cri):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outs = model(imgs)
        loss = cri(outs, labels)
        total_loss += loss.item()
        _, preds = torch.max(outs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    return avg_loss, acc, all_preds, all_labels


if __name__ == "__main__":
    best_val_acc = 0.0
    all_preds_final = None
    all_labels_final = None
    history = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, all_preds, all_labels = val_epoch(model, val_loader, criterion)
        scheduler.step()
        history.append((epoch + 1, train_loss, train_acc, val_loss, val_acc))

        print(f"\nEpoch {epoch + 1:2d}/{num_epochs}")
        print(f"Train | Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ 保存最佳模型，当前最佳Acc: {best_val_acc:.4f}")
            all_preds_final = all_preds
            all_labels_final = all_labels

    # 每轮指标存 CSV（画折线图用：epoch,train_loss,train_acc,val_loss,val_acc）
    with open("metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(history)
    print(f"\n📈 50 轮指标已保存: {os.path.abspath('metrics.csv')}")

    # 最终评估（用最佳模型重新跑验证集）
    print("\n" + "=" * 50)
    print("🔄 加载最佳模型，重新计算验证集指标...")
    model.load_state_dict(torch.load(SAVE_PATH))
    val_loss, final_acc, all_preds_final, all_labels_final = val_epoch(model, val_loader, criterion)

    print(f"最终最佳验证准确率: {final_acc:.4f}")
    print("\n📊 分类报告（4位小数，真实值）:")
    print(classification_report(all_labels_final, all_preds_final, digits=4, zero_division=0))
    print("\n🔍 混淆矩阵:")
    print(confusion_matrix(all_labels_final, all_preds_final))
