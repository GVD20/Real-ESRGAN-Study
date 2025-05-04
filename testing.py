import cv2
import torch
import numpy as np
import os
import yaml
os.environ['TORCHINDUCTOR_DISABLE'] = '1'
from realesrgan.models.realesrgan_model import RealESRGANModel

img = cv2.imread('ex.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img).float() / 255.0  # 转为 [0, 1] 范围
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).cuda()  # 转为 NCHW 格式

# 初始化退化模型
with open('tests/data/get_img_test.yml', mode='r') as f: # 读取配置文件
    opt = yaml.load(f, Loader=yaml.FullLoader)
model = RealESRGANModel(opt)
model.is_train = True  # 设置为训练模式

# 模拟输入数据
data = {
    'gt': img_tensor,  # 高质量图像
    'kernel1': torch.ones(7, 7).float().cuda(),  # 模拟模糊核1
    'kernel2': torch.ones(7, 7).float().cuda(),  # 模拟模糊核2
    'sinc_kernel': torch.ones(7, 7).float().cuda()  # 模拟 sinc 核
}

# 调用退化管线
model.feed_data(data)


# 获取退化后的图像
degraded_img = model.lq.squeeze(0).permute(1, 2, 0).cpu().numpy()
degraded_img = (degraded_img * 255).clip(0, 255).astype(np.uint8)

# 检查退化后的图像值范围和形状
print("model.lq shape:", model.lq.shape)
print("model.lq min:", model.lq.min().item())
print("model.lq max:", model.lq.max().item())

# 检查转换后的图像值范围
print("degraded_img min:", degraded_img.min())
print("degraded_img max:", degraded_img.max())

# 保存退化后的图像
cv2.imwrite('degraded_ex.png', cv2.cvtColor(degraded_img, cv2.COLOR_RGB2BGR))