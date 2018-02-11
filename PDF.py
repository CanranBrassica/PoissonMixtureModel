import numpy as np
import numpy.random as rd
from scipy.special import digamma
import math


class Gamma:
    # Gam(x|k,\theta) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)},
    # k=shape param, 1/\theta=scale param
    def __init__(self, shape, scale):
        self.shape = shape
        self.scale = scale
        self.pdf = rd.gamma(shape=shape, scale=scale)
        self.ex = shape / scale  # <x>
        self.v = shape / scale**2  # <x^2>-<x>^2
        self.ex_ln = digamma(shape) - math.log(scale)  # <ln x>

    def get_sample(self):
        return rd.gamma(shape=self.shape, scale=1 / self.scale)


class Dirichlet:
    # Dir(\vec{x}|\alpha) = C_D(\alpha)\prod_{i=1}^{k}{x^{\alpha_i-1}_i}
    def __init__(self, alpha):
        self.alpha = alpha
        self.ex = np.array([alpha_k / np.sum(alpha) for alpha_k in alpha])  # <x>
        self.ex_ln = np.array([digamma(alpha_k) - digamma(np.sum(alpha)) for alpha_k in alpha])  # <ln x>

    def get_sample(self):
        return rd.dirichlet(alpha=self.alpha)


class Categorical:
    # Cat(\vec{x}|\vec{\pi}) = \prod_{k=1}^{K}{\pi_k^{x_k}
    # pi = distribution param
    def __init__(self, dist):
        self.dist= dist
        self.ex = dist# <x>
        self.ex2 = dist# <x^2>


class Poission:
    # Poi(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!}
    def __init__(self, lam):
        self.lam = lam
        self.ex = lam
        self.ex2 = lam * (lam + 1)

    def get_sample(self):
        return rd.poisson(lam=self.lam)
