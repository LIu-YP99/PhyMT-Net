
from src.config import Config
from src.dataset import DynamicSeismicDataset

# 加载训练集、验证集和测试集
train_ds = DynamicSeismicDataset(split='train', augment=False)
val_ds = DynamicSeismicDataset(split='val', augment=False)
test_ds = DynamicSeismicDataset(split='test', augment=False)

# 统计样本数量
n_train = len(train_ds)
n_val = len(val_ds)
n_test = len(test_ds)

total = n_train + n_val + n_test

print("\n========== Dataset Statistics ==========")
print(f"Training samples:   {n_train}")
print(f"Validation samples: {n_val}")
print(f"Test samples:       {n_test}")
print(f"Total samples:      {total}")

print("\n========== Dataset Proportions ==========")
print(f"Training:   {n_train / total * 100:.2f}%")
print(f"Validation: {n_val / total * 100:.2f}%")
print(f"Test:       {n_test / total * 100:.2f}%")
