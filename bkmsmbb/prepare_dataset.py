import os
import random
import shutil

print("--- 开始划分数据集 ---")

# --- 设置 ---
# 原始数据目录 (你通过爬虫下载的图片存放位置)
SOURCE_DIR = "../dataset_scraped"
# 新的数据集目录 (用于训练)
DEST_DIR = "../dataset"
# 验证集所占的比例
VAL_SPLIT = 0.20

# --- 执行 ---
if not os.path.exists(SOURCE_DIR):
    print(f"错误：原始数据目录 '{SOURCE_DIR}' 不存在。请先运行抓取脚本。")
    exit()

# 清理并创建目标文件夹
if os.path.exists(DEST_DIR):
    print(f"检测到已存在的目标目录 '{DEST_DIR}'，将先删除它...")
    shutil.rmtree(DEST_DIR)

# 创建 train 和 val 文件夹
train_dir = os.path.join(DEST_DIR, "train")
val_dir = os.path.join(DEST_DIR, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print(f"已创建新的数据集目录: '{DEST_DIR}'")

# 遍历每个类别 (pokemon, digimon)
for class_name in os.listdir(SOURCE_DIR):
    class_source_dir = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_source_dir):
        continue

    print(f"\n正在处理类别: {class_name}")

    # 为每个类别创建 train 和 val 子文件夹
    class_train_dir = os.path.join(train_dir, class_name)
    class_val_dir = os.path.join(val_dir, class_name)
    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_val_dir, exist_ok=True)

    # 获取所有图片文件列表并打乱
    images = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]
    random.shuffle(images)

    # 计算分割点
    split_point = int(len(images) * VAL_SPLIT)
    val_images = images[:split_point]
    train_images = images[split_point:]

    # 复制文件到验证集
    for img in val_images:
        shutil.copy(os.path.join(class_source_dir, img), class_val_dir)
    print(f"  > 已复制 {len(val_images)} 张图片到验证集 (val/{class_name})")

    # 复制文件到训练集
    for img in train_images:
        shutil.copy(os.path.join(class_source_dir, img), class_train_dir)
    print(f"  > 已复制 {len(train_images)} 张图片到训练集 (train/{class_name})")

print("\n--- 数据集划分完成！ ---")