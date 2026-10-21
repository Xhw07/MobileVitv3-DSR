import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# ----------------------【原版模型定义，和你train.py保持一致！】----------------------
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.5, proj_drop=0.5):
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
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.5):
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
    def __init__(self, dim, depth, num_heads, mlp_ratio=4, dropout=0.5, sr_ratio=8):
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
    def __init__(self, num_classes=4, dims=[16, 32, 64], depths=[2, 4, 3], num_heads=4, dropout=0.5, sr_ratio=8):
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
# ----------------------模型定义结束----------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileViTv3(num_classes=4, dropout=0.5, sr_ratio=8).to(device)
weight_path = "best_model_reg95.pth"
model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()

IMG_SIZE = 96
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder(
    root=r"D:/科研/智慧城市/dataset_split/test",
    transform=test_transform
)
print("===自动读取数据集类别===")
print("类别顺序（索引0,1,2,3）：", test_dataset.classes)
print("索引映射关系：", test_dataset.class_to_idx)
class_names = test_dataset.classes

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_preds = []
all_labels = []
all_probs = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        out = model(imgs)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(out, dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(prob.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
num_classes = 4

cls_metrics = []
for cls in range(num_classes):
    y_true_bin = (all_labels == cls).astype(int)
    y_pred_bin = (all_preds == cls).astype(int)
    y_prob_bin = all_probs[:, cls]
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    acc_cls = (tp + tn) / (tp + tn + fp + fn)
    prec_cls = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_cls = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_cls = 2 * prec_cls * rec_cls / (prec_cls + rec_cls) if (prec_cls + rec_cls) > 0 else 0.0
    # ==========新增 特异度 Specificity==========
    spec_cls = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc_cls = roc_auc_score(y_true_bin, y_prob_bin)
    cls_metrics.append([acc_cls, prec_cls, rec_cls, f1_cls, spec_cls, auc_cls])

# 打印带真实类别名的表格，增加Specificity列
print(f"\n{'类别':<10}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Specificity':<12}{'AUC':<12}")
print("-"*84)
for idx, (acc, prec, rec, f1, spec, auc) in enumerate(cls_metrics):
    print(f"{class_names[idx]:<10}{acc:<12.4f}{prec:<12.4f}{rec:<12.4f}{f1:<12.4f}{spec:<12.4f}{auc:<12.4f}")

# 全局指标
acc_global = accuracy_score(all_labels, all_preds)
prec_macro = precision_score(all_labels, all_preds, average="macro")
rec_macro = recall_score(all_labels, all_preds, average="macro")
f1_macro = f1_score(all_labels, all_preds, average="macro")
auc_ovr = roc_auc_score(all_labels, all_probs, multi_class="ovr")
# 全局Macro-Specificity：所有类别特异度取平均
spec_list = [m[4] for m in cls_metrics]
spec_macro = np.mean(spec_list)

print("-"*84)
print(f"{'全局Macro':<10}{acc_global:<12.4f}{prec_macro:<12.4f}{rec_macro:<12.4f}{f1_macro:<12.4f}{spec_macro:<12.4f}{auc_ovr:<12.4f}")

print("\n====混淆矩阵====")
print(confusion_matrix(all_labels, all_preds))
