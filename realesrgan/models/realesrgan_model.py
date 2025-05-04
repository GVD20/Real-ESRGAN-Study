import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F


@MODEL_REGISTRY.register()
# 这个类是 Real-ESRGAN 模型的实现，继承自 SRGANModel 类。
class RealESRGANModel(SRGANModel):
    """RealESRGAN 模型用于 Real-ESRGAN: 使用纯合成数据训练真实世界的盲超分辨率。

    它主要执行以下操作：
    1. 在 GPU Tensor Core中随机合成低质量 (LQ) 图像
    2. 使用 GAN 训练优化网络。
    """

    def __init__(self, opt):
        super(RealESRGANModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()  # 模拟 JPEG 压缩
        self.usm_sharpener = USMSharp().cuda()  # 进行usm锐化
        self.queue_size = opt.get('queue_size', 180) # 训练对池的大小

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """这是一个用于增加批次中多样性的训练对池。

        批处理限制了批次中合成退化的多样性。例如，一个批次中的样本可能无法具有不同的缩放因子。
        因此，我们使用这个训练对池来增加批次中的退化多样性。

        总之就是可以在训练中使用参数不同的低质量 (LQ) 图像来增加多样性。

        输入和输出的形状如下：
        - 输入：
            - lq: (b, c, h, w) 低质量图像
            - gt: (b, c, h, w) 高质量图像

        - 输出：
            - lq: (b, c, h, w) 低质量图像
            - gt: (b, c, h, w) 高质量图像

        在过程中，函数对输入的低质量图像进行入队，并从队列中出队一个低质量图像和一个高质量图像。
        """
        # initialize 初始化
        b, c, h, w = self.lq.size() # 批次大小，通道，高度，宽度
        if not hasattr(self, 'queue_lr'): # 如果没有队列低质量图像
            # 断言队列大小是否可以被批处理大小整除
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda() #使用pytorch调用cuda创建一个全零的张量，写入队列的低质量图像
            _, c, h, w = self.gt.size() # 写入队列的高质量图像
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda() # 使用pytorch调用cuda创建一个全零的张量，写入队列的高质量图像
            self.queue_ptr = 0 #队列指针，指向下一个要写入的位置
        if self.queue_ptr == self.queue_size:  # 队列满了
            # do dequeue and enqueue 进行出队和入队
            # shuffle 打乱
            idx = torch.randperm(self.queue_size) # 使用pytorch的随机排列函数打乱队列
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples 获取第一个 b 个样本
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue 更新队列
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue # 将出队的低质量图像赋值给当前的低质量图像
            self.gt = gt_dequeue # 将出队的高质量图像赋值给当前的高质量图像
        else:
            # only do enqueue 进行入队
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone() # 将当前的低质量图像赋值给队列
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()# 将当前的高质量图像赋值给队列
            self.queue_ptr = self.queue_ptr + b # 队列指针加上批次大小

    @torch.no_grad()
    def feed_data(self, data): # 喂数据
        """从数据加载器接受数据，然后添加两级退化以获得低质量 (LQ) 图像。
        """
        # 如果是训练模式，并且高阶退化选项为真，则进行高阶退化
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis 训练数据合成

            self.gt = data['gt'].to(self.device) # 将待退化图像扔到GPU上

            # 进行usm锐化。USM是一种常见的锐化办法。具体步骤是：
            # 1. 使用高斯模糊滤波器对图像进行模糊处理，得到模糊图像。
            # 2. 将模糊图像与原图像进行差分，得到高频图像。
            # 3. 将高频图像与原图像进行加权平均，得到锐化图像。
            # 4. 将锐化图像与原图像进行加权平均，得到最终的锐化图像。
            self.gt_usm = self.usm_sharpener(self.gt) # USM锐化

            self.kernel1 = data['kernel1'].to(self.device) # 第一个模糊核
            self.kernel2 = data['kernel2'].to(self.device) # 第二个模糊核
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            ori_h, ori_w = self.gt.size()[2:4]

            # 两次退化区别在于第二次退化的模糊核和噪声范围不同，其他的都是一样的。
            # 具体来说，第二次退化的模糊核是一个更小的核，噪声范围是一个更小的范围。
            # 这样做的目的是为了让第二次退化的图像更接近真实世界中的图像。

            # ----------------------- 第一个退化流程 ----------------------- #
            # 模糊 blur
            out = filter2D(self.gt_usm, self.kernel1)
            # 随机缩放 random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0] # 随机选择缩放类型
            # up: 放大 down: 缩小 keep: 保持原尺寸
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic']) # 随机选择插值方式
            # area: 区域插值 bilinear: 双线性插值 bicubic: 三次插值
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            out = out / (self.kernel1.size(0) * self.kernel1.size(1)) # 除以模糊核的大小
            # 添加噪声 add noise
            gray_noise_prob = self.opt['gray_noise_prob'] # 随机选择添加噪声的概率
            # 添加高斯噪声或泊松噪声
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                print("Used Gaussian noise")
            else:
                out = random_add_poisson_noise_pt( # 添加泊松噪声
                    out,
                    scale_range=self.opt['poisson_scale_range'], # 泊松噪声的范围
                    gray_prob=gray_noise_prob, # 添加灰度噪声的概率
                    clip=True,
                    rounds=False)
                print("Used Poisson noise")
            # JEPG压缩 JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range']) # 随机选择JPEG压缩的质量
            # 这里的jpeg_p是一个张量，表示每个图像的JPEG压缩质量
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts 钳制到 [0, 1]，否则 JPEGer 会搞出不好的伪影
            out = self.jpeger(out, quality=jpeg_p)


            # ----------------------- 第二个退化流程 ----------------------- #
            # 模糊 blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
            # 随机缩放 random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
                print("Used upsampling")
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
                print("Used downsampling")
            else:
                scale = 1
                print("Used keep")
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
            out = out / (self.kernel2.size(0) * self.kernel2.size(1))
            # 添加噪声 add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                print("Used Gaussian noise")
            else:
                print("Used Poisson noise")
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
                    # JPEG压缩 + 最终的sinc滤波器
                    # 我们还需要将图像调整为所需的尺寸。我们将 [调整回原尺寸 + sinc滤波器] 组合为一个操作。
                    # 我们考虑两种顺序：
                    #   1. [调整回原尺寸 + sinc滤波器] + JPEG压缩
                    #   2. JPEG压缩 + [调整回原尺寸 + sinc滤波器]
                    # 根据经验，我们发现其他组合（sinc + JPEG + 调整尺寸）会引入扭曲的线条。
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                print("Used sinc filter first")
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = out / (self.sinc_kernel.size(0) * self.sinc_kernel.size(1))
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                print("Used sinc filter second")
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                out = out / (self.sinc_kernel.size(0) * self.sinc_kernel.size(1))
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                out = filter2D(out, self.sinc_kernel)
            # clamp and round 钳制并取整
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop 随机裁剪
            gt_size = self.opt['gt_size']
            (self.gt, self.gt_usm), self.lq = paired_random_crop([self.gt, self.gt_usm], self.lq, gt_size,
                                                                 self.opt['scale'])

            # training pair pool 训练对池
            # 这里的训练对池是一个用于存储训练对的队列。我们将当前的低质量图像和高质量图像添加到队列中。
            # 如果队列满了，我们就从队列中取出一个低质量图像和高质量图像。
            self._dequeue_and_enqueue()
            # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
            # 再次锐化 self.gt，因为我们已经用 self._dequeue_and_enqueue 更改了 self.gt
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract 警告：grad 和 param 不遵守梯度布局约定
        # 不是训练模式，或者高阶退化选项为假，则直接使用数据加载器提供的数据
        else:
            # for paired training or validation 成对训练或验证
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    # 这个函数用于验证模型的性能。它会在验证集上运行模型，并计算损失。
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation 不要在验证期间使用合成过程
        self.is_train = False
        super(RealESRGANModel, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    # 这个函数用于优化模型的参数。它会计算损失，并更新模型的参数。
    def optimize_parameters(self, current_iter):
        # usm sharpening usm 锐化
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
        if self.opt['l1_gt_usm'] is False:
            l1_gt = self.gt
        if self.opt['percep_gt_usm'] is False:
            percep_gt = self.gt
        if self.opt['gan_gt_usm'] is False:
            gan_gt = self.gt

        # optimize net_g 优化 net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss 像素损失
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss 感知损失
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d 优化 net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real 真
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        l_d_real.backward()
        # fake 假
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
        l_d_fake.backward()
        self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
