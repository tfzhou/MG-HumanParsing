import functools
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from inplace_abn.bn import InPlaceABNSync

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


class GumbleSoftmax(torch.nn.Module):
    """
    Gumbel Softmax Sampler, Requires 2D input [batchsize, number of categories]
    Does not support single binary category. Use two dimensions with softmax instead.
    """

    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard

    # def sample_gumbel(self, shape, eps=1e-10):
    #     """Sample from Gumbel(0, 1)"""
    #     noise = torch.rand(shape)
    #     noise.add_(eps).log_().neg_()
    #     noise.add_(eps).log_().neg_()
    #     return Variable(noise).cuda()

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(-1)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + Variable(gumble_samples_tensor)
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)
        return soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args: logits: [batch_size, n_class] unnormalized log-probs
              temperature: non-negative scalar
              hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns: [batch_size, n_class] sample from the Gumbel-Softmax distribution.

        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            _, max_value_indexes = y.data.max(1, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y

    def forward(self, logits, temp=1, force_hard=False):

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True)


class GatingBlock(nn.Module):
    """Gating mechanism, inputs: [B, C, H, W]"""

    def __init__(self, in_dim, out_dim, force_hard):
        super(GatingBlock, self).__init__()
        self.force_hard = force_hard
        # Gate layers
        self.fc1 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.fc1_bn = BatchNorm2d(out_dim)
        self.fc2 = nn.Conv2d(out_dim, 2, kernel_size=1)
        self.gs = GumbleSoftmax().cuda()

    def forward(self, x, temperature=1):
        # Compute relevance score
        # w = F.avg_pool2d(x, x.size(2))
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1_bn(self.fc1(w)))
        w = self.fc2(w)
        # Sample from Gumble Module
        w = self.gs(w, temp=temperature, force_hard=self.force_hard)

        out = x * w[:, 1].unsqueeze(1)
        # Return output of layer and the value of the gate
        # The value of the gate will be used in the target rate loss
        return out, w[:, 1]


class SpatialDropout(nn.Module):
    """Dropout pixel values using custom shape"""

    def __init__(self, drop_prob):

        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, inputs):
        output = inputs.clone()
        if not self.training or self.drop_prob == 0:
            return inputs
        else:
            noise = self._make_noise(inputs)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(inputs)
            output.mul_(noise)
        return output

    def _make_noise(self, input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))
