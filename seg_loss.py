import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')


class SoftDiceWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceWithLogitsLoss, self).__init__()
 
    def forward(self, logits, targets, smooth=1, p=2):
        probs = F.sigmoid(logits)
        m1 = probs.view(-1)
        m2 = targets.view(-1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum() + smooth) / (m1.pow(p).sum() + m2.pow(p).sum() + smooth)
        score = 1 - score
        return score

class BatchSoftDiceWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BatchSoftDiceWithLogitsLoss, self).__init__()
 
    def forward(self, logits, targets, smooth=1e-3, p=2):
        batch_size = logits.shape[0]
        probs = F.sigmoid(logits)
        m1 = probs.view(batch_size,-1)
        m2 = targets.view(batch_size,-1)
        intersection = (m1 * m2) # (b,length)
 
        # score = 2. * (intersection.sum(1) + smooth) / (m1.pow(p).sum(1) + m2.pow(p).sum(1) + smooth) # (b,)
        score = (2. * intersection.sum(1) + smooth) / (m1.pow(p).sum(1) + m2.pow(p).sum(1) + smooth) # (b,)

        score = 1 - score
        return score.mean()

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, probs, targets, smooth=1, p=2):
        # m1 = probs.view(-1)
        # m2 = targets.view(-1)
        intersection = torch.sum(probs * targets)
 
        score = 2. * (intersection + smooth) / (probs.pow(p).sum() + targets.pow(p).sum() + smooth)
        score = 1 - score
        return score





def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: input data tensor with shape :math:`(N, 1, *)`.
        target: the target tensor with shape :math:`(N, 1, *)`.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: for numerically stability when dividing.
    Returns:
        the computed loss.
    Examples:
        >>> num_classes = 1
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[[6.325]]],[[[5.26]]],[[[87.49]]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(4.6052)
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    probs = torch.sigmoid(input)

    # avoid inf or nan
    probs = torch.clamp(probs, min=1e-3, max=0.999)

    if probs.shape != target.shape:
        target = target.reshape(probs.shape) 
    
    loss_tmp = -alpha * torch.pow((1.0 - probs + eps), gamma) * target * torch.log(probs + eps) - (
        1 - alpha
    ) * torch.pow(probs + eps, gamma) * (1.0 - target) * torch.log(1.0 - probs + eps)

    loss_tmp = loss_tmp.squeeze(dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none') -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(input, target, self.alpha, self.gamma, self.reduction, self.eps)


