import math

import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions import Uniform
from torch.distributions import Exponential
from torch.distributions import Cauchy

from scipy.stats import levy_stable

from modules import baseline_network
from modules import glimpse_network, core_network
from modules import action_network, location_network


class RecurrentAttention(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """
    def __init__(self,
                 g,
                 k,
                 s,
                 c,
                 h_g,
                 h_l,
                 std,
                 hidden_size,
                 num_classes,
                 config):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse.
        - s: scaling factor that controls the size of successive patches.
        - c: number of channels in each image.
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the rnn.
        - num_classes: number of classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()
        # RNN type
        self.rnn_type = config.rnn_type;

        # when the locations l is defined by a Gaussian distribution
        self.std = std

        # when the locations l is defined by a symmetry stable distribution
        self.alpha = config.alpha 
        self.gamma = config.gamma


        self.sensor = glimpse_network(h_g, h_l, g, k, s, c, config)
        self.rnn = core_network(hidden_size, hidden_size, config)
        self.locator = location_network(hidden_size, 2, config)
        self.classifier = action_network(hidden_size, num_classes)
        self.baseliner = baseline_network(hidden_size, 1)

    def forward(self, x, l_t_prev, h_t_prev, last=False, cell_state_prev=None):
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, C, H, W). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a vector of length (B,). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        - log_pi: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)

        if self.rnn_type == 'RNNCell':
            h_t = self.rnn(g_t, h_t_prev)
        elif self.rnn_type == 'LSTMCell':
            h_t, cell_state = self.rnn(g_t, h_t_prev, cell_state_prev)

        mu, l_t = self.locator(h_t, l_t_prev) #mu, l_t = self.locator(h_t)
        b_t = self.baseliner(h_t).squeeze()

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs

        # normal
        #log_pi = Normal(mu, self.std).log_prob(l_t)
        #log_pi = torch.sum(log_pi, dim=1)

        # stable distribution
        # use scipy, too slow
        #log_pi = levy_stable.logpdf(mu.cpu().detach().numpy(), self.alpha, 0, loc = 0, scale = self.gamma) # this is too slow for gpu
        #log_pi = torch.sum(torch.tensor(log_pi, dtype = torch.float32, device = mu.device, requires_grad = True), dim=1)
        
        # use CMS method to simulate random variable sampled form stalbe distribution; it is a random variable not a pdf
        # zeta = 0 # self.beta * torch.tan(pi * self.alpha / 2)
        # epson = (1 / self.alpha) * torch.atan(-zeta)
        # U = Uniform(-math.pi/2,math.pi/2)
        # E = Exponential(1)
        # X = ((1 + zeta**2)**(1/2/self.alpha)) * torch.sin(self.alpha * (U + epson)) / (torch.cos(U)**(1/self.alpha)) * (torch.cos(U - self.alpha * (U + epson)) / E)**((1 - self.alpha)/self.alpha)
        # Y = self.gamma * X;
        # log_pi = torch.log(Y)
        # log_pi = torch.sum(log_pi, dim=1)

        # use cauchy to temporally replace it
        log_pi = Cauchy(loc = mu, scale = self.gamma).log_prob(l_t - l_t_prev) # to do: use mu + noise to replace l_t
        log_pi = torch.sum(log_pi, dim=1)

        if self.rnn_type == 'RNNCell':
            if last:
              log_probas = self.classifier(h_t)
              return h_t, l_t, b_t, log_probas, log_pi
            else:
              return h_t, l_t, b_t, log_pi
        elif self.rnn_type == 'LSTMCell':
            if last:
              log_probas = self.classifier(h_t)
              return h_t, l_t, b_t, log_probas, log_pi, cell_state
            else:
              return h_t, l_t, b_t, log_pi, cell_state