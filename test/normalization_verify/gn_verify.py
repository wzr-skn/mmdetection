import numpy as np
import torch.nn as nn
import torch


def group_norm(x: torch.Tensor,
               num_groups: int,
               num_channels: int,
               eps: float = 1e-5,
               gamma: float = 1.0,
               beta: float = 0.):
    assert divmod(num_channels, num_groups)[1] == 0
    channels_per_group = num_channels // num_groups

    new_tensor = []
    for t in x.split(channels_per_group, dim=1):
        var_mean = torch.var_mean(t, dim=[1, 2, 3], unbiased=False)
        var = var_mean[0]
        mean = var_mean[1]
        t = (t - mean[:, None, None, None]) / torch.sqrt(var[:, None, None, None] + eps)
        t = t * gamma + beta
        new_tensor.append(t)

    new_tensor = torch.cat(new_tensor, dim=1)
    return new_tensor


def main():
    num_groups = 2
    num_channels = 4
    eps = 1e-5

    img = torch.rand(2, num_channels, 2, 2)
    print(img)

    gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)
    r1 = gn(img)
    print(r1)

    r2 = group_norm(img, num_groups, num_channels, eps)
    print(r2)

if __name__ == '__main__':
    main()
