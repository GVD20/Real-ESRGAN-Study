import os
import random
import shutil

# 定义路径
src_dir = "testDataSet/Endovis18"  # 源图像目录
dest_base = "testDataSet/Endovis18_Shunt"  # 目标基础目录
train_dir = os.path.join(dest_base, "train")  # 训练集目录
run_dir = os.path.join(dest_base, "run")  # 运行集目录
train_txt = os.path.join(dest_base, "train.txt")  # 训练文件列表

# 创建目标目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(run_dir, exist_ok=True)

# 获取源目录中的所有图片文件
images = []
for root, _, files in os.walk(src_dir):
    for file in files:
        # 假设图片扩展名为常见图片格式
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            images.append(os.path.join(root, file))

# 随机打乱文件列表
random.shuffle(images)

# 按1:9的比例分割
split_index = len(images) // 10  # 1/10作为训练集
train_images = images[:split_index]
run_images = images[split_index:]

# 存储训练图片的目标路径
train_dest_paths = []

# 复制文件到目标目录
print(f"复制 {len(train_images)} 个文件到训练集...")
for img in train_images:
    rel_path = os.path.relpath(img, src_dir)
    dst_path = os.path.join(train_dir, rel_path)

    # 确保目标子目录存在
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # 复制文件
    shutil.copy2(img, dst_path)

    # 记录目标路径
    train_dest_paths.append(dst_path)

print(f"复制 {len(run_images)} 个文件到运行集...")
for img in run_images:
    rel_path = os.path.relpath(img, src_dir)
    dst_path = os.path.join(run_dir, rel_path)

    # 确保目标子目录存在
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # 复制文件
    shutil.copy2(img, dst_path)

# 创建训练集文件列表
print("创建训练集文件列表...")
with open(train_txt, 'w') as f:
    for path in train_dest_paths:
        # 将路径格式调整为需要的格式
        standard_path = os.path.join("train",
                                    os.path.relpath(path, train_dir))
        # 确保路径使用正斜杠
        standard_path = standard_path.replace('\\', '/')
        f.write(f"{standard_path}\n")

print(f"处理完成! 训练集: {len(train_images)}张, 运行集: {len(run_images)}张")
print(f"训练集文件列表已保存至: {train_txt}")