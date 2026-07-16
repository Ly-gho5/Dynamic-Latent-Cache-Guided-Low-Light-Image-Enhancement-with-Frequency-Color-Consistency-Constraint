import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
import kornia
import os
import math
import time
import matplotlib.pyplot as plt
from basicsr.models.losses.loss_util import weighted_loss
# from basicsr.metrics.psnr_ssim import calculate_psnr
# from kornia.constants import pi
_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')
@weighted_loss
def smooth_l1_loss(pred, target):
    return F.smooth_l1_loss(pred, target, reduction='mean', beta=1.47/2.)
# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)

rgb_from_hed = torch.tensor([[0.65, 0.70, 0.29],
                            [0.07, 0.99, 0.11],
                            [0.27, 0.57, 0.78]])
# hed_from_rgb = linalg.inv(rgb_from_hed)
hed_from_rgb = torch.linalg.inv(rgb_from_hed)


def separate_stains(rgb, conv_matrix):
    # rgb = _prepare_colorarray(rgb, force_copy=True, channel_axis=-1)
    # rgb = rgb.astype(np.float32)

    # rgb = torch.maximum(rgb, torch.tensor(1e-6))  # avoiding log artifacts
    rgb = torch.clamp(rgb, min=1e-6)
    log_adjust = torch.log(torch.tensor(1e-6))  # used to compensate the sum above
    rgb = rearrange(rgb, 'b c h w -> b h w c')
    stains = (torch.log(rgb) / log_adjust) @ conv_matrix
    stains = rearrange(stains, 'b h w c -> b c h w')
    # stains = torch.maximum(stains, torch.tensor(0.))
    stains = torch.clamp(stains, min=0.)
    return stains

def combine_stains(stains, conv_matrix):

    # stains = stains.astype(np.float32)

    # log_adjust here is used to compensate the sum within separate_stains().
    log_adjust = -torch.log(torch.tensor(1e-6))
    stains = rearrange(stains, 'b c h w -> b h w c')
    log_rgb = -(stains * log_adjust) @ conv_matrix
    rgb = torch.exp(log_rgb)
    rgb = rearrange(rgb, 'b h w c -> b c h w')
    return torch.clamp(rgb, min=0., max=1.)

class rgb2hed(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mat = hed_from_rgb

    def forward(self, x):
        return separate_stains(x, self.mat.to(x.device))
class hed2rgb(nn.Module):
    def __init__(self,):
        super().__init__()
        self.mat = rgb_from_hed

    def forward(self, x):
        return combine_stains(x, self.mat.to(x.device))
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += l1_loss(
                predi, target, weight, reduction=self.reduction)
            return self.loss_weight * loss
        else:
            return self.loss_weight * l1_loss(
                pred, target, weight, reduction=self.reduction)

class ClassifyLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ClassifyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, pred, gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # print(pred.shape, gt.shape)

        return self.loss_weight * self.loss(pred, gt.squeeze(-1))
class ClassifyLossTrain(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ClassifyLossTrain, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        # self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # print(pred.shape, gt.shape)

        return self.loss_weight * self.loss(pred, gt)
class FocusDistanceL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FocusDistanceL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, distance_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += l1_loss(
                predi, distance_gt, weight, reduction=self.reduction)
            return self.loss_weight * loss
        else:
            return self.loss_weight * l1_loss(
                pred, distance_gt, weight, reduction=self.reduction)
class FocusDistanceSmoothL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FocusDistanceSmoothL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, distance_gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += smooth_l1_loss(
                predi, distance_gt, weight, reduction=self.reduction)
            return self.loss_weight * loss
        else:
            return self.loss_weight * smooth_l1_loss(
                pred, distance_gt, weight, reduction=self.reduction)
class L1LossPry(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1LossPry, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        loss = 0.
        target_pry = kornia.geometry.build_pyramid(target, len(pred))
        loss += l1_loss(
            pred[-1], target, weight, reduction=self.reduction)
        pred.pop(-1)
        for i, predi in enumerate(pred):
            loss += l1_loss(
            predi, target_pry[i], weight, reduction=self.reduction) * 0.33

        return self.loss_weight * loss

class ApeLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ApeLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.MSELoss()

    def forward(self, predict, hr):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        # print(predict)
        sr, ic = predict['img'], predict['ics']
        pre_psnr = 0
        for sr_i, ic_i in zip(sr, ic):
            now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
            # print(now_psnr.shape, sr_i.shape, hr.shape)
            ic_i_gt = 1 - torch.tanh(now_psnr - pre_psnr)
            pre_psnr = now_psnr
            # print('ic_i_gt: ', ic_i_gt.shape, ic_i.shape, ic_i.T.squeeze().shape)
            # loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.T.squeeze(), ic_i_gt)
            loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.squeeze(-1), ic_i_gt)
            # print(self.diff_loss(sr_i, hr), self.mse_loss(ic_i.squeeze(-1), ic_i_gt))
        # loss = 0.1 * loss + self.diff_loss(sr[-1], hr)
        # print('loss', loss)
        return loss
class ApeV2Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ApeV2Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.MSELoss()

    def forward(self, predict, hr):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        # print(predict)
        sr, ic = predict['img'], predict['ics']
        pre_psnr = 10.0 * torch.log10(1. ** 2 / ((sr[0] - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
        loss = loss + self.diff_loss(sr[0], hr)
        for sr_i, ic_i in zip(sr[1:], ic):
            now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
            # print(now_psnr.shape, sr_i.shape, hr.shape)
            ic_i_gt = 1 - torch.tanh(now_psnr - pre_psnr)
            # ic_i_gt = torch.tanh(now_psnr - pre_psnr)
            pre_psnr = now_psnr
            # print('ic_i_gt: ', ic_i_gt.shape, ic_i.shape, ic_i.T.squeeze().shape)
            # loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.T.squeeze(), ic_i_gt)
            loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.squeeze(-1), ic_i_gt)
            # print('ics: ', ic_i.squeeze(-1), 'ics_gt: ', ic_i_gt, 'psnr: ', pre_psnr)
            # print('divide: ', self.diff_loss(sr_i, hr) / self.mse_loss(ic_i.squeeze(-1), ic_i_gt))
        # loss = 0.1 * loss + self.diff_loss(sr[-1], hr)
        # print('loss', loss)
        return loss
class AdaptiveFreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptiveFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.MSELoss()
        self.gamma = 10.

    def forward(self, predict, hr):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        # print(predict)
        sr, ic = predict['img'], predict['ics']
        pre_psnr = 10.0 * torch.log10(1. ** 2 / ((sr[0] - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
        # loss = loss + self.diff_loss(sr[0], hr)
        for sr_i, ic_i in zip(sr[1:-1], ic):
            now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
            # print(now_psnr.shape, sr_i.shape, hr.shape)
            ic_i_gt = torch.tanh((now_psnr - pre_psnr) * self.gamma)
            # ic_i_gt = torch.tanh(now_psnr - pre_psnr)
            pre_psnr = now_psnr
            # print('ic_i_gt: ', ic_i_gt.shape, ic_i.shape, ic_i.T.squeeze().shape)
            # loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.T.squeeze(), ic_i_gt)
            loss = loss + self.mse_loss(ic_i.squeeze(-1), ic_i_gt) # * 10.
            # print('ics: ', ic_i.squeeze(-1), 'ics_gt: ', ic_i_gt, 'psnr: ', pre_psnr)
            # print('divide: ', self.diff_loss(sr_i, hr) / self.mse_loss(ic_i.squeeze(-1), ic_i_gt))
        # print('loss', loss, self.diff_loss(sr[-1], hr))
        loss = loss + self.diff_loss(sr[-1], hr) #  * 0.1
        # print('loss', loss)
        return loss
class AdaptiveLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(AdaptiveLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        # self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.L1Loss() # nn.MSELoss()
        self.gamma = 1. # 5. # 10.
    def forward(self, predict, hr):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        # print(predict)
        sr, ic = predict['img'], predict['ics']
        pre_psnr = 10.0 * torch.log10(1. ** 2 / ((sr[0] - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)

        for sr_i, ic_i in zip(sr[1:], ic):
            now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
            # print(now_psnr.shape, sr_i.shape, hr.shape)
            # ic_i_gt = torch.tanh((now_psnr - pre_psnr) * self.gamma)
            ic_i_gt = (now_psnr - pre_psnr) * self.gamma
            # ic_i_gt = torch.tanh(now_psnr - pre_psnr)
            pre_psnr = now_psnr
            # print('ic_i_gt: ', ic_i_gt.shape, ic_i.shape, ic_i.T.squeeze().shape)
            # loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i.T.squeeze(), ic_i_gt)
            loss = loss + self.mse_loss(ic_i.squeeze(-1), ic_i_gt) # * 10.
            # print('ics: ', ic_i.squeeze(-1), 'ics_gt: ', ic_i_gt)
            # print('divide: ', self.diff_loss(sr_i, hr) / self.mse_loss(ic_i.squeeze(-1), ic_i_gt))
        # loss = 0.1 * loss + self.diff_loss(sr[-1], hr)
        # print('loss', loss)
        return self.loss_weight * loss / len(ic)
class ApeLossOld(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ApeLossOld, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.diff_loss = FreqLoss(loss_weight, reduction)
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        # dict_list = ['enc_0', 'enc_1', 'enc_2', 'mid_3', 'dec_2', 'dec_1', 'dec_0']
        loss = 0.
        target_pry = kornia.geometry.build_pyramid(target, 5)
        pred_images_dict = pred[0]
        dict_list = pred_images_dict.keys()
        ics_dict = pred[1]
        for i in dict_list:
            k = int(i.split('_')[-1])
            pre_psnr = 0.
            sr, ic = pred_images_dict[i], ics_dict[i]
            hr = target_pry[k]
            for sr_i, ic_i in zip(sr, ic):
                now_psnr = 10.0 * torch.log10(1. ** 2 / ((sr_i - hr) ** 2).mean(dim=(1, 2, 3)) + 1e-8)
                # kornia.metrics.psnr(sr_i, hr, 1.)
                # print(now_psnr)
                ic_i_gt = 1 - torch.tanh(now_psnr - pre_psnr)
                # print(ic_i_gt.squeeze().shape, ic_i.shape)
                # print(now_psnr, ic_i_gt, self.mse_loss(ic_i, ic_i_gt))
                pre_psnr = now_psnr
                # print(ic_i.shape, ic_i_gt, sr_i.shape, now_psnr)
                loss = loss + self.diff_loss(sr_i, hr) + self.mse_loss(ic_i, ic_i_gt.squeeze())
        loss = 0.1 * loss + self.diff_loss(pred[-1], target)
        # print(loss)
        return loss
class ShiftLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ShiftLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                diff = torch.fft.rfft2(predi) - torch.fft.rfft2(target)
                loss_freq = torch.mean(torch.abs(diff))
                loss += self.loss_weight * loss_freq
            return loss
        else:
            p_w = torch.fft.rfft(pred, dim=-1)
            t_w = torch.fft.rfft(target, dim=-1)
            # diff = (p_w * torch.abs(t_w)) / (t_w * torch.abs(p_w))
            diff = torch.angle(t_w) - torch.angle(p_w)

            loss = torch.mean(torch.abs(diff))
            # print(loss)
            return self.loss_weight * loss * 0.05
class FreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                diff = torch.fft.rfft2(predi) - torch.fft.rfft2(target)
                loss_freq = torch.mean(torch.abs(diff))
                loss += self.loss_weight * (loss_freq * 0.01 + self.l1_loss(predi, target))
            return loss / len(pred)
        else:
            diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
            loss = torch.mean(torch.abs(diff))
            # print(loss)
            return self.loss_weight * (loss * 0.01 + self.l1_loss(pred, target))
def FReLU(img):
    x = torch.fft.rfft2(img)
    x_real = torch.relu(x.real)
    x_imag = torch.relu(x.imag)
    x = torch.complex(x_real, x_imag)
    x = torch.fft.irfft2(x) - img / 2.
    return x
class FRLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FRLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                diff = FReLU(predi) - FReLU(target)
                loss_freq = torch.mean(torch.abs(diff))
                loss += self.loss_weight * loss_freq * 4. + self.l1_loss(predi, target)
            return loss
        else:
            diff = FReLU(pred) - FReLU(target)
            loss = torch.mean(torch.abs(diff))
            # print(loss, self.l1_loss(pred, target))
            return self.loss_weight * loss * 4. + self.l1_loss(pred, target)
class SelfFRLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SelfFRLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.window_size = 6
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = CharbonnierLoss(loss_weight, reduction)

    def forward(self, pred, target, lr):
        if isinstance(pred, list):
            loss = 0.
            target_ = [lr, target, target]
            for i, predi in enumerate(pred):
                x = torch.fft.rfft2(predi)
                x.real = torch.relu(x.real)
                x.imag = torch.relu(x.imag)
                x = torch.fft.irfft2(x) - predi / 2.
                x = torch.fft.fftshift(x, dim=[-2, -1])
                x_center = kornia.geometry.center_crop(x, [self.window_size, self.window_size])
                loss_freq = -(torch.log(torch.mean(torch.abs(x_center))+1e-5))
                y = torch.fft.rfft2(target_[i])
                y.real = torch.relu(y.real)
                y.imag = torch.relu(y.imag)
                y = torch.fft.irfft2(y) - target_[i] / 2.
                y = torch.fft.fftshift(y, dim=[-2, -1])
                # loss_freq = -(torch.log(torch.mean(torch.abs(x-y)) + 1e-5)) + loss_freq_c
                loss_freq = torch.clamp(loss_freq, -1., 1.)
                L_reg = torch.abs(torch.mean(x)-torch.mean(y))
                L1 = self.l1_loss(predi, target_[i])
                # print(loss_freq * 1., L_reg * 1000., L1) # * 0.01
                loss += self.loss_weight * loss_freq*0.1 + L_reg * 10. # + L1*0.05
            return loss
        else:
            x = torch.fft.rfft2(pred)
            x.real = torch.relu(x.real)
            x.imag = torch.relu(x.imag)
            x = torch.fft.irfft2(x) - pred / 2.
            x = torch.fft.fftshift(x, dim=[-2, -1])
            x_center = kornia.geometry.center_crop(x, [self.window_size, self.window_size])
            loss_freq = -torch.mean(torch.abs(x_center))
            y = torch.fft.rfft2(target)
            y.real = torch.relu(y.real)
            y.imag = torch.relu(y.imag)
            y = torch.fft.irfft2(y) - target / 2.
            y = torch.fft.fftshift(y, dim=[-2, -1])
            L_reg = torch.abs(torch.mean(x - y))
            L1 = self.l1_loss(pred, target)
            loss = self.loss_weight * loss_freq * 0.01 + L_reg * 1000. + L1*5
            # print(loss)
            return loss

class CharFreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CharFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = CharbonnierLoss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target)
class StainLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(StainLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
        self.rgb2hed = rgb2hed()

    def forward(self, pred, target):
        hed = self.rgb2hed(pred)
        Hem, Eos, DAB = torch.chunk(hed, 3, dim=1)
        hedx = self.rgb2hed(target)
        Hemx, Eosx, DABx = torch.chunk(hedx, 3, dim=1)
        HE = torch.cat([Hem, Eos], dim=1)
        HEx = torch.cat([Hemx, Eosx], dim=1)
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * loss * 0.01 + self.l1_loss(pred, target) + self.l1_loss(HE, HEx)
class PhasefreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PhasefreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        pred_angle = torch.angle(pred_f)
        tar_f = torch.fft.rfft2(target)
        phase_diff = torch.abs(torch.angle(tar_f) - pred_angle)
        loss = torch.mean(phase_diff)
        return self.loss_weight * loss * 0.01 # + self.l1_loss(pred, target)
class STNPhaseOnlyLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(STNPhaseOnlyLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        pred_angle = torch.angle(pred_f)
        tar_f = torch.fft.rfft2(target)
        tar_angle = torch.angle(tar_f)
        phase_diff = torch.abs(torch.mean(torch.fft.irfft2(torch.exp(1j*tar_angle))) -
                               torch.mean(torch.fft.irfft2(torch.exp(1j*pred_angle))))
        loss = torch.mean(phase_diff)
        return self.loss_weight * loss * 0.01 # + self.l1_loss(pred, target)
class Phase2freqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Phase2freqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        tar_f = torch.fft.rfft2(target)
        pred_angle = torch.angle(pred_f)
        tar_angle = torch.angle(tar_f)
        # pred_mag = torch.abs(pred_f)
        p_f = torch.exp(1j * pred_angle) # pred_mag * torch.exp(1j * pred_angle)
        t_f = torch.exp(1j * tar_angle) # pred_mag * torch.exp(1j * tar_angle)
        pred_if = torch.fft.irfft2(p_f)
        target_if = torch.fft.irfft2(t_f)
        # var = torch.var(pred_if, dim=1)
        phase_diff = torch.abs(p_f - t_f)
        # diff = torch.abs(tar_f-pred_f)
        loss = torch.mean(phase_diff)
        return (self.l1_loss(pred_if, target_if) + loss * 0.05) * self.loss_weight# self.loss_weight * loss * 0.01 +
class Phase3freqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(Phase3freqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss2 = Phase2freqLoss(loss_weight, reduction)
    def forward(self, pred, target):
        pred1, pred2 = pred

        return torch.mean(torch.abs(pred1)) + self.freq_loss2(pred2, target)# self.loss_weight * loss * 0.01 +
class PANSRFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PANSRFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss = FreqLoss(loss_weight, reduction)
    def forward(self, pred, target, lr):
        loss = 0.
        # for predi in pred:
        loss += self.freq_loss(pred[0], lr)
        loss += self.freq_loss(pred[1], target)
        loss += self.freq_loss(pred[2], target)
        return loss
class PANSR_ShiftFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(PANSR_ShiftFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss = FreqLoss(loss_weight, reduction)
        self.shift_loss = ShiftLoss(loss_weight, reduction)
    def forward(self, pred, target, lr):
        loss = 0.
        # for predi in pred:
        loss += self.freq_loss(pred[0], lr) + self.shift_loss(pred[0], lr)
        loss += self.freq_loss(pred[1], target) + self.shift_loss(pred[1], target)
        loss += self.freq_loss(pred[2], target) + self.shift_loss(pred[2], target)
        return loss
class MultiFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MultiFreqLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        # self.loss_weight = loss_weight
        # self.reduction = reduction
        # self.freq_loss1 = Phase2freqLoss(loss_weight, reduction)
        self.freq_loss = FreqLoss(loss_weight, reduction)
    def forward(self, pred, target):
        loss = 0.
        for predi in pred:
            loss += self.freq_loss(predi, target)
        return loss
class MultiScaleFreqLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.freq_loss = FreqLoss(loss_weight, reduction)

    def forward(self, pred, target):
        tar = target
        loss = 0.
        if isinstance(pred, list):
            for predi in pred[::-1]:
                loss += self.freq_loss(predi, tar)
                tar = F.interpolate(tar, scale_factor=0.5)
        else:
            loss += self.freq_loss(pred, tar)
        return loss
class FocalfreqsinLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FocalfreqsinLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)
    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred)
        tar_f = torch.fft.rfft2(target)
        diff = pred_f - tar_f
        phase_diff = torch.abs(torch.angle(tar_f) - torch.angle(pred_f))
        # phase_diff = phase_diff / (2 * 3.1415926536)
        phase_diff = torch.sin(phase_diff / 2.)
        loss = torch.mean(torch.abs(diff) * 0.01 + torch.pow(phase_diff, 2) * 0.1)
        return self.loss_weight * loss + self.l1_loss(pred, target)
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        if isinstance(pred, list):
            loss = 0.
            for pre in pred:
                loss += self.forward_single(pre, target)
            return loss / len(pred)
        else:
            return self.forward_single(pred, target)

    def forward_single(self, pred, target):

        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss




class CosineLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super(CosineLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        # [B, ...] -> [B, N]
        input_flat = input.reshape(input.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)

        # 正确按 flatten 后的向量做归一化
        input_norm = F.normalize(input_flat, dim=1, eps=self.eps)
        target_norm = F.normalize(target_flat, dim=1, eps=self.eps)

        # [B]
        cos_sim = (input_norm * target_norm).sum(dim=1)
        cos_loss = 1.0 - cos_sim

        if self.reduction == 'mean':
            return cos_loss.mean()
        elif self.reduction == 'sum':
            return cos_loss.sum()
        else:
            return cos_loss


class WHTBlock(nn.Module):
    def __init__(
        self,
        block_size=16,
        thresh=0.1,
        calc_iwht=True,
        isdiff=True,
        final_level='wht',
        normalized=True,
        updown_flg=False
    ):
        super(WHTBlock, self).__init__()
        self.block_size = block_size
        self.normalized = normalized
        self.calc_iwht = calc_iwht
        self.threshold = thresh
        self.isdiff = isdiff
        self.final_level = final_level
        self.updown_flg = updown_flg

        walsh = self.generate_walsh_matrix(block_size).float()
        self.register_buffer("walsh_matrix", walsh)

    @staticmethod
    def generate_walsh_matrix(n):
        assert (n & (n - 1)) == 0, "n 必须是 2 的幂次"
        if n == 1:
            return torch.ones((1, 1), dtype=torch.float32)
        h = WHTBlock.generate_walsh_matrix(n // 2)
        return torch.cat([
            torch.cat([h, h], dim=1),
            torch.cat([h, -h], dim=1)
        ], dim=0)

    def wht_1d(self, x):
        n = x.shape[-1]
        H = self.walsh_matrix.to(device=x.device, dtype=x.dtype)
        if self.normalized:
            H = H / math.sqrt(n)
        return torch.matmul(x, H.T)

    def iwht_1d(self, x):
        n = x.shape[-1]
        H = self.walsh_matrix.to(device=x.device, dtype=x.dtype)
        if self.normalized:
            H = H.T / math.sqrt(n)
        else:
            H = H.T / n
        return torch.matmul(x, H)

    def wht_2d(self, x):
        x = self.wht_1d(x)
        x = self.wht_1d(x.transpose(-1, -2)).transpose(-1, -2)
        return x

    def iwht_2d(self, x):
        x = self.iwht_1d(x)
        x = self.iwht_1d(x.transpose(-1, -2)).transpose(-1, -2)
        return x

    @staticmethod
    def _merge_blocks(block_tensor):
        b, c, nH, nW, bs1, bs2 = block_tensor.shape
        return block_tensor.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, nH * bs1, nW * bs2)

    @staticmethod
    def _to_show_img(x, force_gray=False, log_scale=False, use_abs=False, vis_idx=0):
        x = x[vis_idx].detach().float().cpu()

        if log_scale:
            x = torch.log1p(torch.abs(x))
        elif use_abs:
            x = torch.abs(x)

        if x.dim() == 3:
            c = x.shape[0]
            if (c == 3) and (not force_gray):
                img = x.permute(1, 2, 0).numpy()
            elif c == 1:
                img = x[0].numpy()
            else:
                img = x.mean(dim=0).numpy()
        else:
            img = x.numpy()

        img_min = img.min()
        img_max = img.max()
        if img_max - img_min < 1e-12:
            img = np.zeros_like(img)
        else:
            img = (img - img_min) / (img_max - img_min)

        return img

    def _save_single_image(self, img, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(4, 4))
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _show_debug(
        self,
        debug_dict,
        title_prefix="pred",
        vis_idx=0,
        show_debug=True,
        save_debug=False,
        save_dir=None,
        step=None
    ):
        """
        保存/显示：
        - 原图
        - WHT 系数图
        - 上/下分量系数图
        - 全重建图 / 上下分量重建图
        - 重建误差图
        """
        show_items = [
            ("original",    "original",      False, False, False),
            ("full_wht",    "wht_coeff",     True,  True,  False),
            ("up_wht",      "wht_up_coeff",  True,  True,  False),
            ("down_wht",    "wht_down_coeff",True,  True,  False),
            ("full_recon",  "iwht_recon",    False, False, False),
            ("up_recon",    "iwht_up_recon", False, False, False),
            ("down_recon",  "iwht_down_recon", False, False, False),
            ("recon_error", "recon_error",   True,  False, True),
        ]

        # 构造保存目录
        save_root = None
        if save_debug and save_dir is not None:
            if step is None:
                step_name = time.strftime("%Y%m%d_%H%M%S")
            else:
                step_name = f"step_{step}"
            save_root = os.path.join(save_dir, step_name, title_prefix)
            os.makedirs(save_root, exist_ok=True)

        # 先准备所有图，既可保存单图，也可做总图
        rendered_imgs = []
        rendered_titles = []

        for key, name, force_gray, log_scale, use_abs in show_items:
            if key not in debug_dict:
                rendered_imgs.append(None)
                rendered_titles.append(name)
                continue

            img = self._to_show_img(
                debug_dict[key],
                force_gray=force_gray,
                log_scale=log_scale,
                use_abs=use_abs,
                vis_idx=vis_idx
            )
            rendered_imgs.append(img)
            rendered_titles.append(name)

            # 保存单图
            if save_root is not None:
                save_path = os.path.join(save_root, f"{name}.png")
                self._save_single_image(img, save_path)

        # 保存或显示总图
        if show_debug or (save_root is not None):
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            axes = axes.flatten()

            for ax, img, name in zip(axes, rendered_imgs, rendered_titles):
                if img is None:
                    ax.axis("off")
                    continue

                if img.ndim == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
                ax.set_title(f"{title_prefix} - {name}")
                ax.axis("off")

            plt.tight_layout()

            if save_root is not None:
                grid_path = os.path.join(save_root, f"{title_prefix}_summary.png")
                plt.savefig(grid_path, bbox_inches='tight', dpi=200)

            if show_debug:
                plt.show()
            else:
                plt.close()

    def _pad_and_unfold(self, input):
        b, c, h, w = input.shape
        block_size = self.block_size
        h_pad = ((h + block_size - 1) // block_size) * block_size - h
        w_pad = ((w + block_size - 1) // block_size) * block_size - w
        input_ = F.pad(input, (0, w_pad, 0, h_pad), 'reflect')
        input_blocks = input_.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        return input_, input_blocks, h, w

    def cal_wht(self, input, threshold, return_debug=False):
        input_pad, input_blocks, h, w = self._pad_and_unfold(input)
        x = self.wht_2d(input_blocks)
        x_coeff = torch.abs(x)

        x_out_list = []
        recon_list = []

        if isinstance(threshold, (int, float)):
            x_copy = x.clone()
            x_copy[x_coeff < threshold] = 0
            x_out_list.append(x_copy)
            if return_debug:
                recon_list.append(self.iwht_2d(x_copy))

        elif isinstance(threshold, (np.ndarray, list, torch.Tensor)):
            if isinstance(threshold, torch.Tensor):
                threshold = threshold.detach().cpu().numpy()

            for t in threshold:
                x_copy = x.clone()
                x_copy[x_coeff < t] = 0
                x_out_list.append(x_copy)
                if return_debug:
                    recon_list.append(self.iwht_2d(x_copy))
        else:
            raise ValueError("Threshold should be either a number or an array.")

        if not return_debug:
            return x_out_list

        first_recon_blocks = recon_list[0]
        first_recon = self._merge_blocks(first_recon_blocks)[..., :h, :w]
        full_wht_img = self._merge_blocks(x)
        thresh_wht_img = self._merge_blocks(x_out_list[0])

        debug_dict = {
            "original": input,
            "full_wht": full_wht_img,
            "up_wht": thresh_wht_img,
            "full_recon": first_recon,
            "recon_error": torch.abs(input - first_recon),
        }

        return x_out_list, debug_dict

    def cal_wht_2d_diff(self, input, threshold, isdiff=True, final_level='ori', return_debug=False):
        b, c, h, w = input.shape
        input_pad, input_blocks, h, w = self._pad_and_unfold(input)

        x = self.wht_2d(input_blocks)
        x_coeff = torch.abs(x)
        y_copy_list = []

        first_x_copy = None
        first_y_copy = None

        if isinstance(threshold, (int, float)):
            x_copy = x.clone()
            x_copy[x_coeff < threshold] = 0
            y_copy = self.iwht_2d(x_copy)

            first_x_copy = x_copy
            first_y_copy = y_copy

            if isdiff:
                y_copy_list.append(torch.abs(input_blocks - y_copy))
            else:
                y_copy_list.append(y_copy)

        elif isinstance(threshold, (np.ndarray, list, torch.Tensor)):
            if isinstance(threshold, torch.Tensor):
                threshold = threshold.detach().cpu().numpy()

            for idx, t in enumerate(threshold):
                x_copy = x.clone()
                x_copy[x_coeff < t] = 0
                y_copy = self.iwht_2d(x_copy)

                if idx == 0:
                    first_x_copy = x_copy
                    first_y_copy = y_copy

                if isdiff:
                    y_copy_list.append(torch.abs(input_blocks - y_copy))
                else:
                    y_copy_list.append(y_copy)
        else:
            raise ValueError("Threshold should be either a number or an array.")

        if final_level == 'ori':
            y_copy_list.append(input_blocks)
        elif final_level == 'wht':
            y_copy_list.append(x)

        if not return_debug:
            return y_copy_list

        first_recon = self._merge_blocks(first_y_copy)[..., :h, :w]
        debug_dict = {
            "original": input,
            "full_wht": self._merge_blocks(x),
            "up_wht": self._merge_blocks(first_x_copy),
            "full_recon": first_recon,
            "recon_error": torch.abs(input - first_recon),
        }

        return y_copy_list, debug_dict

    def call_half_wht(self, input, return_debug=False):
        b, c, h, w = input.shape
        input_pad, input_blocks, h, w = self._pad_and_unfold(input)

        x = self.wht_2d(input_blocks)

        x_coeff_up = x.clone()
        x_coeff_down = x.clone()

        x_coeff_down[:, :, :, :, :self.block_size // 2, :self.block_size // 2] = 0
        x_coeff_up[:, :, :, :, self.block_size // 2:, :self.block_size // 2] = 0

        y_up = self.iwht_2d(x_coeff_up)
        y_down = self.iwht_2d(x_coeff_down)
        y_full = self.iwht_2d(x)

        y_copy_list = [y_up, y_down, x]

        if not return_debug:
            return y_copy_list

        full_wht_img = self._merge_blocks(x)
        up_wht_img = self._merge_blocks(x_coeff_up)
        down_wht_img = self._merge_blocks(x_coeff_down)

        up_recon = self._merge_blocks(y_up)[..., :h, :w]
        down_recon = self._merge_blocks(y_down)[..., :h, :w]
        full_recon = self._merge_blocks(y_full)[..., :h, :w]

        debug_dict = {
            "original": input,
            "full_wht": full_wht_img,
            "up_wht": up_wht_img,
            "down_wht": down_wht_img,
            "full_recon": full_recon,
            "up_recon": up_recon,
            "down_recon": down_recon,
            "recon_error": torch.abs(input - full_recon),
        }

        return y_copy_list, debug_dict

    def forward(
        self,
        input,
        debug=False,
        debug_name="input",
        vis_idx=0,
        show_debug=True,
        save_debug=False,
        save_dir=None,
        step=None
    ):
        if self.updown_flg:
            if debug:
                outlist, debug_dict = self.call_half_wht(input, return_debug=True)
                self._show_debug(
                    debug_dict,
                    title_prefix=debug_name,
                    vis_idx=vis_idx,
                    show_debug=show_debug,
                    save_debug=save_debug,
                    save_dir=save_dir,
                    step=step
                )
                return outlist
            else:
                return self.call_half_wht(input, return_debug=False)

        if self.calc_iwht:
            if debug:
                outlist, debug_dict = self.cal_wht(input, self.threshold, return_debug=True)
                self._show_debug(
                    debug_dict,
                    title_prefix=debug_name,
                    vis_idx=vis_idx,
                    show_debug=show_debug,
                    save_debug=save_debug,
                    save_dir=save_dir,
                    step=step
                )
                return outlist
            else:
                return self.cal_wht(input, self.threshold, return_debug=False)
        else:
            if debug:
                outlist, debug_dict = self.cal_wht_2d_diff(
                    input=input,
                    threshold=self.threshold,
                    isdiff=self.isdiff,
                    final_level=self.final_level,
                    return_debug=True
                )
                self._show_debug(
                    debug_dict,
                    title_prefix=debug_name,
                    vis_idx=vis_idx,
                    show_debug=show_debug,
                    save_debug=save_debug,
                    save_dir=save_dir,
                    step=step
                )
                return outlist
            else:
                return self.cal_wht_2d_diff(
                    input=input,
                    threshold=self.threshold,
                    isdiff=self.isdiff,
                    final_level=self.final_level,
                    return_debug=False
                )

##################################################################################################################################enhanced loss

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
class CosineLoss_new(nn.Module):
    def __init__(self, reduction='mean', eps=1e-8):
        super(CosineLoss_new, self).__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        input_flat = input.reshape(input.size(0), -1)
        target_flat = target.reshape(target.size(0), -1)

        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        input_norm = F.normalize(input_flat, dim=-1, eps=self.eps)
        target_norm = F.normalize(target_flat, dim=-1, eps=self.eps)
        cos_sim = (input_norm * target_norm).sum(dim=-1)
        cos_loss = 1.0 - cos_sim  # Cosine Distance

        if self.reduction == 'mean':
            return cos_loss.mean()
        elif self.reduction == 'sum':
            return cos_loss.sum()
        else:
            return cos_loss




class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        C1 = 0.01**2
        C2 = 0.03**2
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)
    


#####################################################################################################针对颜色改进的loss
def rgb_to_ycbcr(x: torch.Tensor):
    """
    x: [B,3,H,W], 
    返回: Y, Cb, Cr  
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y)
    cr = 0.713 * (r - y)
    return y, cb, cr


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sqrt((x - y) ** 2 + self.eps ** 2)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HueCosineLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        # pred/gt: [B,3,H,W]
        pred_v = pred / (pred.norm(dim=1, keepdim=True) + self.eps)
        gt_v   = gt   / (gt.norm(dim=1, keepdim=True)   + self.eps)

        cos = (pred_v * gt_v).sum(dim=1, keepdim=True).clamp(-1, 1)  # [B,1,H,W]
        hue_loss = 1.0 - cos

        # 用 GT luminance 加权，避免纯黑区域颜色向量不稳定
        y, _, _ = rgb_to_ycbcr(gt)
        w = (y.detach().clamp(0, 1) ** 0.5)  # sqrt 权重，别让暗处完全没梯度
        return (hue_loss * w).mean()


class GlobalColorMomentLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        # [B,3,H,W] -> per-image per-channel stats
        mu_p = pred.mean(dim=(2, 3))  # [B,3]
        mu_g = gt.mean(dim=(2, 3))

        var_p = pred.var(dim=(2, 3), unbiased=False) + self.eps
        var_g = gt.var(dim=(2, 3), unbiased=False) + self.eps
        std_p = var_p.sqrt()
        std_g = var_g.sqrt()

        return F.l1_loss(mu_p, mu_g) + F.l1_loss(std_p, std_g)


class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size(1) * t.size(2) * t.size(3)

class FSPC(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 lambda_value=(1, 2, 1, 1),
                 w_pix=0.05,         # 像素域保色（很小即可，不破坏去噪）
                 w_chroma=0.25,      # 色度校正（Cb/Cr）
                 w_hue=0.10,         # 色相对齐
                 w_moment=0.05,      # 全局白平衡/色调
                 use_charb=True,
                 reduction="mean",
                 **kwargs):

        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.lambda_value = list(lambda_value)

        self.block_size = 32
        self.thresh = [0.2, 0.1, 0.05]
        self.wht = WHTBlock(block_size=self.block_size,
                            thresh=self.thresh,
                            calc_iwht=True,
                            isdiff=True,
                            final_level='wht',
                            normalized=True,
                            updown_flg=True)

        self.l1 = nn.L1Loss(reduction=reduction)
        self.base_diff = CharbonnierLoss() if use_charb else nn.L1Loss(reduction=reduction)


        self.hue_loss = HueCosineLoss()
        self.moment_loss = GlobalColorMomentLoss()
        self.w_pix = w_pix
        self.w_chroma = w_chroma
        self.w_hue = w_hue
        self.w_moment = w_moment

    def forward(self, pred, gt, weight=None, **kwargs):
        pred_list = self.wht(pred)
        gt_list   = self.wht(gt)
        assert len(pred_list) == len(gt_list)

        loss_sasw = 0.0

        for i in range(len(pred_list) - 1):
            loss_sasw = loss_sasw + self.lambda_value[i] * self.base_diff(pred_list[i], gt_list[i])

        loss_sasw = loss_sasw + self.lambda_value[-1] * self.l1(pred_list[-1], gt_list[-1])

        loss_pix = self.l1(pred, gt)

        _, cb_p, cr_p = rgb_to_ycbcr(pred)
        _, cb_g, cr_g = rgb_to_ycbcr(gt)
        loss_chroma = self.base_diff(cb_p, cb_g) + self.base_diff(cr_p, cr_g)

        loss_hue = self.hue_loss(pred, gt)

        loss_moment = self.moment_loss(pred, gt)

        loss = loss_sasw \
               + self.w_pix * loss_pix \
               + self.w_chroma * loss_chroma \
               + self.w_hue * loss_hue \
               + self.w_moment * loss_moment

        return loss*self.loss_weight


