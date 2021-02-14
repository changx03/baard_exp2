import torch


class AddGaussianNoise(torch.nn.Module):
    """Add Gaussian Noise to a tensor"""

    def __init__(self, mean=0., std=1., eps=0.025, x_min=0., x_max=1.):
        super().__init__()
        self.mean = mean
        self.std = std
        self.eps = eps
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        x_noisy = x + self.eps * (torch.randn(x.size()) * self.std + self.mean)
        x_noisy = torch.clip(x_noisy, self.x_min, self.x_max)
        return x_noisy

    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={}, eps={})'.format(self.mean, self.std, self.eps)
