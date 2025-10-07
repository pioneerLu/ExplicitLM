import torch


class RMSNorm(torch.nn.Module):
    """
    RMS归一化层（Root Mean Square Layer Normalization）

    使用均方根进行归一化，相比LayerNorm计算更简单高效。
    公式: y = (x / RMS(x)) * weight
    其中 RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: 特征维度
        eps: 数值稳定性的小常数，防止除零错误
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算RMS归一化

        Args:
            x: 输入张量

        Returns:
            归一化后的张量
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            归一化并缩放后的张量
        """
        return self.weight * self._norm(x.float()).type_as(x)