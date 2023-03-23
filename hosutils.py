import torch
import pandas as pd
import numpy as np
from functools import partial


eps = 1e-12


def k_sample_moment(x, k, dim=-1):
    return torch.mean(torch.pow(x, k), dim=dim)


def k_central_sample_moment(x, k, dim=-1):
    mu = torch.mean(x, dim=dim, keepdim=True)
    return k_sample_moment(x-mu, k, dim=dim)


def sqrt_k_central_sample_moment(x, k, dim=-1):
    temp = k_central_sample_moment(x, k, dim)
    return torch.pow(temp, 1/k)


def sample_skewness(x, dim=-1, unbiased=True):
    m3 = k_central_sample_moment(x, 3, dim=dim)
    m2 = k_central_sample_moment(x, 2, dim=dim)
    g1 = m3 / (torch.pow(m2, 3/2)+eps)
    n = x.shape[dim]
    if unbiased:
        assert n > 2
        g1_unbiased = ((n*(n-1))**0.5) * g1 / (n - 2)
        return g1_unbiased
    else:
        return g1


def sample_kurtosis(x, dim=-1, unbiased=True):
    m4 = k_central_sample_moment(x, 4, dim=dim)
    m2 = k_central_sample_moment(x, 2, dim=dim)
    g2 = m4 / (torch.pow(m2, 2)+eps) - 3
    n = x.shape[dim]
    if unbiased:
        assert n > 3
        g2_unbiased = (n-1)/((n-2)*(n-3))*((n+1)*g2+6)
        return g2_unbiased
    else:
        return g2


mean = torch.mean
std = torch.std
var = torch.var
skew = sample_skewness
kurt = sample_kurtosis
csm3 = partial(k_central_sample_moment, k=3)
csm4 = partial(k_central_sample_moment, k=4)
scsm3 = partial(sqrt_k_central_sample_moment, k=3)
scsm4 = partial(sqrt_k_central_sample_moment, k=4)
scsm5 = partial(sqrt_k_central_sample_moment, k=5)
scsm6 = partial(sqrt_k_central_sample_moment, k=6)

if __name__ == '__main__':
    x = np.random.randn(10).astype(np.float32)
    xd = pd.Series(x)
    x1 = xd.skew(axis=0)
    x2 = xd.kurt(axis=0)
    xt = torch.from_numpy(x)
    xt1 = sample_skewness(xt)
    xt2 = sample_kurtosis(xt)
    print(x1, x2)
    print(xt1.item(), xt2.item())
