import math

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Modify standard PyTorch distributions so they are compatible with this code.
"""


class TanhNormal(torch.distributions.Distribution):
    def __init__(self, mean, std, epsilon=1e-6):
        super(TanhNormal, self).__init__()
        self.normal_mean = mean
        self.normal_std = std
        self.epsilon = epsilon
        self.normal_dist = torch.distributions.Normal(mean, std)

    def rsample(self, return_pre=False):
        z = self.normal_mean + self.normal_std * torch.distributions.Normal(
            torch.zeros_like(self.normal_mean), torch.ones_like(self.normal_std)).sample()
        if return_pre:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            pre_tanh_value = torch.atanh(value)
        return self.normal_dist.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )
#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()

