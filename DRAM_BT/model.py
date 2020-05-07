import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.distributions import Uniform
from torch.distributions import Exponential
from torch.distributions import Cauchy


from scipy.stats import levy_stable
import cv2

from modules import baseline_network, context_network
from modules import glimpse_network, core_network
from modules import action_network, location_network
from modules import Levy_bottom_up_generator, combine_location_network


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
                 c,
                 image_size,                 
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
        - c: number of channels in each image.   
        - image_size: a tuple: (H x W)     
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the rnn.
        - num_classes: number of classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentAttention, self).__init__()        

        # when the locations l is defined by a Gaussian distribution
        self.std = std

        # when the locations l is defined by a symmetry stable distribution
        self.alpha = config.alpha 
        self.gamma = config.gamma

        self.config = config        
        
        self.context = context_network(c, config.kernel_size, hidden_size)
        self.sensor = glimpse_network(hidden_size, g, c, config)
        self.rnn = core_network(hidden_size, hidden_size, config)
        self.top_down_locator = location_network(hidden_size, 2, config)
        self.bot_up_locator = Levy_bottom_up_generator(config.batch_size, image_size, config)
        self.combine_location = combine_location_network(hidden_size, config)
        self.classifier = action_network(hidden_size, num_classes)
        self.baseliner = baseline_network(hidden_size, 1)
       
        # something for initialzing subroutine
        dtype = (
            torch.cuda.FloatTensor if self.config.use_gpu else torch.FloatTensor
          ) 

        # derivative of Saliecy map
        self.derivative_y = torch.tensor([-1,0,1]).reshape(1,1,3,1).type(dtype)
        self.derivative_x = torch.t(torch.tensor([-1,0,1])).reshape(1,1,1,3).type(dtype)
        # a weighted saliency s gauged at a fixation center
        self.gaussian_kernel_sigma = math.floor(image_size[0] / 12) # in the paper, /6 but pytorch does not accept such big kernel
        gaussian_kernel_size = self.gaussian_kernel_sigma * 2 + 1
        tmp_x, tmp_y = torch.meshgrid(torch.arange(-self.gaussian_kernel_sigma,self.gaussian_kernel_sigma + 1).type(dtype), torch.arange(-self.gaussian_kernel_sigma,self.gaussian_kernel_sigma + 1).type(dtype))       
        self.gaussian_kernel = torch.exp(- (tmp_x.type(dtype)**2 + tmp_y.type(dtype)**2)/self.gaussian_kernel_sigma**2).reshape(1, 1, gaussian_kernel_size, gaussian_kernel_size)
        

    def forward(self, x, l_t_prev, h_t_prev1, h_t_prev2, cell_state_prev1, cell_state_prev2, SM, SM_local_smooth, last=False):
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
        - cell_state_prev: a 2D tensor of shape (B, hidden_size). The gate stale of LSTM 
          for the previous timestep `t-1`.
        - SM: saliency maps B x 1 x H x W
        - SM_local_smooth: local smoothed SM B x 1 x H x W
        - dSMx, dSMy, partial derivative of SM, B x 1 x H x W        
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
        
        h_t1, cell_state1, h_t2, cell_state2 = self.rnn(g_t, h_t_prev1, h_t_prev2, cell_state_prev1, cell_state_prev2)

        b_t = self.baseliner(h_t2).squeeze()

        mu, l_t_top_down = self.top_down_locator(h_t2) #mu, l_t = self.locator(h_t, l_t_prev)

        #with torch.no_grad():
        l_t_bot_up, SM_local_smooth = self.bot_up_locator(SM_local_smooth)

        l_t = self.combine_location(l_t_bot_up,l_t_top_down)

        # we assume both dimensions are independent
        # 1. pdf of the joint is the product of the pdfs
        # 2. log of the product is the sum of the logs

        #normal
        log_pi = Normal(mu, self.std).log_prob(l_t_top_down)
        log_pi = torch.sum(log_pi, dim=1)

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

        # # use cauchy to temporally replace it
        # log_pi = Cauchy(loc = mu, scale = self.gamma).log_prob(l_t - l_t_prev) # to do: use mu + noise to replace l_t
        # log_pi = torch.sum(log_pi, dim=1)
        
        if last:
          log_probas = self.classifier(h_t1)
          return h_t1, h_t2, l_t, b_t, log_probas, log_pi, cell_state1, cell_state2, SM_local_smooth
        else:
          return h_t1, h_t2, l_t, b_t, log_pi, cell_state1, cell_state2, SM_local_smooth

    def initialize(self, x, SM):
        """
        (1) Generate saliency maps, derivatives, and smoothed saliency map, and find the first bottom-up location
        (2) initialize the hidden state 2 as output of context network, and use it to
        initialize the top-down location.
        (3) combine top-down and bottom-up locations to get the final location

        Args
        ----
        - x: a 4D Tensor of shape (B, C, H, W). The minibatch
          of images.        

        Returns
        -------
        - h_t2: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.        
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.  
        - SM: saliency maps B x 1 x H x W
        - dSMx, dSMy, partial derivative of SM, B x 1 x H x W
        - SM_local_smooth: local smoothed SM
        """    
        # saliency map related
        B = x.shape[0] 
        
        # smooth with periodic boundary conditions
        r_pad_SM = nn.ReflectionPad2d(int(self.gaussian_kernel_sigma))
        SM_pad = r_pad_SM(SM)
        SM_local_smooth = F.conv2d(SM_pad, self.gaussian_kernel)  

        # normalize to [0,1]
        SM_local_smooth = SM_local_smooth - torch.min(SM_local_smooth.reshape(B,-1),dim=1)[0].reshape(B,1,1,1)
        SM_local_smooth = SM_local_smooth / torch.max(SM_local_smooth.reshape(B,-1),dim=1)[0].reshape(B,1,1,1) 

        # h_t2 and l_t
        dtype = (
            torch.cuda.FloatTensor if self.config.use_gpu else torch.FloatTensor
        )        
        
        # initialize hidden state 1 as output of context network
        h_t2 = self.context(x)

        # use the hidden state to generate the first top-down location
        _, l_t_top_down = self.top_down_locator(h_t2) #mu, l_t = self.locator(h_t2, l_t_prev)


        # initialize the first location as the center of image to bottom-up locator to generate the first location used in the model                
        #with torch.no_grad():
        l_t_bot_up, SM_local_smooth = self.bot_up_locator(SM_local_smooth)

        l_t = self.combine_location(l_t_bot_up,l_t_top_down)

        return h_t2, l_t, SM_local_smooth
