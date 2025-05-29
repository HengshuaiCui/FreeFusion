import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def Y_Upper(y1, y2, hyp_prm=1.3):
    C = 0.0001
    # μ_yk
    y1_mean = torch.mean(y1)
    y2_mean = torch.mean(y2)
    # y_k upperwave
    y1_mean_sub = y1 - y1_mean
    y2_mean_sub = y2 - y2_mean
    # c_k and c^(c_upperArrow)
    c1 = torch.norm(y1_mean_sub)
    c2 = torch.norm(y2_mean_sub)
    c_upperArrow = torch.maximum(c1, c2)
    # wmygfh
    c_upperArrow *= hyp_prm

    # s_k
    s1 = y1_mean_sub / (c1 + C)
    s2 = y2_mean_sub / (c2 + C)

    # s upper dash
    s_upperDash = s1 + s2
    # s_upperDash = tf.maximum(s1, s2)
    # s^
    s_upperArrow = s_upperDash / (torch.norm(s_upperDash) + C)

    # y^
    y_upperArrow = c_upperArrow * s_upperArrow

    return y_upperArrow


def _th_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=0)
    x_data = np.expand_dims(x_data, axis=0)

    y_data = np.expand_dims(y_data, axis=0)
    y_data = np.expand_dims(y_data, axis=0)

    x = torch.from_numpy(x_data)
    y = torch.from_numpy(y_data)

    g = torch.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / torch.sum(g)


def th_SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = _th_fspecial_gauss(size, sigma).cuda()  # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = F.conv2d(img1, window)  # 均值1
    mu2 = F.conv2d(img2, window)  # 均值2
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window) - mu1_sq  # 方差1
    sigma2_sq = F.conv2d(img2 * img2, window) - mu2_sq  # 方差2
    sigma12 = F.conv2d(img1 * img2, window) - mu1_mu2  # 协方差

    value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    value = value.mean()
    return 1 - value


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()

    def forward(self, ir_image, visimage_bri, res_weight, fus_img, upper_weight, mean_weight):
        target = Y_Upper(ir_image, visimage_bri, upper_weight).cuda()
        ssim_loss = th_SSIM_LOSS(target, fus_img)
        pre_loss = abs(torch.mean(mean_weight - torch.sum(res_weight, dim=1, keepdim=False)))
        return ssim_loss + pre_loss, [ssim_loss, pre_loss, pre_loss]
