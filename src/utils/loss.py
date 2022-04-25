import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyVotingLoss(nn.Module):
    """
    Proxy voting loss, reference paper from [6DoF Object Pose Estimation via Differentiable Proxy Voting Loss](https://arxiv.org/pdf/2002.03923.pdf).
    """
    def __init__(self):
        super().__init__()

    def forward(self, out, target, reduction='sum'):
        r"""
        Proxy voting loss, reference paper from [6DoF Object Pose Estimation via Differentiable Proxy Voting Loss](https://arxiv.org/pdf/2002.03923.pdf).

        :param out: shape is (B, H, W, K, 2), the vector field without normalize
        :param target: shape is (B, H, W, K, 2), the vector field without normalize
        :param reduction: 'sum' or 'mean'
        :return: proxy voting loss

        .. math::
            L_{pv} = \sum_{k \in K}\sum_{p \in M} {\mathit{l_1}}(\frac{|v_k^y k^x-v_k^x k^y+v_k^x p^y-v_k^y p^x|}{\sqrt{(v_k^x)^2+(v_k^y)^2}})

        >>> proxy_loss = ProxyVotingLoss()
        >>> out = torch.ones((1, 2, 3, 2, 2))
        >>> target = torch.ones((1, 2, 3, 2, 2))
        >>> proxy_loss(out, target, 'mean').item()
        0.0
        >>> out = torch.zeros((1, 2, 3, 2, 2))
        >>> target = torch.ones((1, 2, 3, 2, 2))
        >>> proxy_loss(out, target, 'sum').item()
        0.0
        >>> out = torch.zeros((1, 2, 3, 2, 2)) + 2
        >>> target = torch.ones((1, 2, 3, 2, 2))
        >>> proxy_loss(out, target, 'sum').item()
        0.0
        >>> out = torch.tensor([[[[0.0, 1.0], [0, 0]]]]).view(1, 1, 2, 1, 2)
        >>> target = torch.tensor([[[[1.0, 0], [0, 0]]]]).view(1, 1, 2, 1, 2)
        >>> proxy_loss(out, target, 'sum').item()
        0.5
        """
        out_norm = torch.clamp_min(torch.norm(out, p=2, dim=-1), 1e-6)
        loss_part = out[:, :, :, :, 1] * target[:, :, :, :, 0] - out[:, :, :, :, 0] * target[:, :, :, :, 1]
        loss = torch.abs(loss_part) / out_norm
        return F.smooth_l1_loss(loss, torch.zeros_like(loss), reduction=reduction)
