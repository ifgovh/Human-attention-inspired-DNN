import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np
import math
import cv2

class retina(object):
    """
    A retina that extracts a foveated glimpse `phi`
    around location `l` from an image `x`. It encodes
    the region around `l` at a high-resolution but uses
    a progressively lower resolution for pixels further
    from `l`, resulting in a compressed representation
    of the original image `x`.

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch Batch, Channels, Height, Width
      of images.
    - l: a 2D Tensor of shape (B, 2). Contains normalized
      coordinates in the range [-1, 1].
    - g: size of the first square patch.
      successive patches.

    Returns
    -------
    - phi: a 4D tensor of shape (B, C, g, g). The
      foveated glimpse of the image.
    """
    def __init__(self, g, config):
        self.g = g
        self.use_gpu = config.use_gpu

 #    when you need extrach multiple patches from one glimpse like RAM,
 #    but you shoule make the size of the output same as the conv layer
 #    def foveate(self, x, l):
 #        """
 #        Extract `k` square patches of size `g`, centered
 #        at location `l`. The initial patch is a square of
 #        size `g`, and each subsequent patch is a square
 #        whose side is `s` times the size of the previous
 #        patch.

 #        The `k` patches are finally resized to (g, g) and
 #        concatenated into a tensor of shape (B, k, g, g, C).
 #        """
 #        phi = []
 #        size = self.g

 #        # extract k patches of increasing size
 #        for i in range(self.k):
 #            phi.append(self.extract_patch(x, l, size))
 #            size = int(self.s * size)
        
 #        # resize the patches to squares of size g
 #        for i in range(1, len(phi)):
 #            k = phi[i].shape[-1] // self.g
 #            phi[i] = F.avg_pool2d(phi[i], k)

 #        # concatenate into a single tensor and flatten        
 #        phi = torch.cat(phi, 1)
 #        import pdb;pdb.set_trace()
 #        phi = phi.view(phi.shape[0], -1)
        
 #        return phi

    def extract_patch(self, x, l):
        """
        Extract a single patch for each image in the
        minibatch `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, C, H, W). The minibatch
          of images.
        - l: a 2D Tensor of shape (B, 2).        

        Returns
        -------
        - patch: a 4D Tensor of shape (B, C, size, size)
        """
        size = self.g
        B, C, H, W = x.shape
        # calculate coordinate for each batch samle (padding considered)
        from_x, from_y = l[:, 0], l[:, 1]

        # build fluid-flow grid
        if self.use_gpu:
            theta = torch.cuda.FloatTensor(B*2,3).fill_(0)
        else:
            theta = torch.zeros(B*2,3)

        # see onenote of affine transform for this algorithm (Pytorch theta is different with cv2's affine matrix)
        theta[torch.arange(0,B*2,2),0] = size / W
        theta[torch.arange(1,B*2,2),1] = size / H
        theta[torch.arange(0,B*2,2),2] = from_x
        theta[torch.arange(1,B*2,2),2] = from_y
        theta = theta.reshape((B,2,3))
              
        grid = F.affine_grid(theta, torch.Size((B,C,size,size)))
        
        return F.grid_sample(x,grid,mode='nearest',padding_mode='zeros') #padding_mode='reflection'    


class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    three conv layers, following a fc layer. The glimpse location 
    vector `l_t_prev` to a fc layer. These two fc layers have the same
    size. Finally, their element-wise mutiplication is rectified.

    In other words:

        `g_t = relu( fc( l ) + fc( 3conv2d(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`. image patch;
      it equse the hidden size
    - g: size of the square patches in the glimpses extracted
      by the retina.
    - c: number of channels in each image.c
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """
    def __init__(self, h_g, g, c, config):
        super(glimpse_network, self).__init__()
        self.retina = retina(g, config)

        # glimpse layer
        padding_size = int((config.kernel_size-1)*0.5)# make sure the output size is the same as input size. 
        self.conv1 = nn.Conv2d(c,c,config.kernel_size,padding=padding_size)
        self.conv2 = nn.Conv2d(c,c,config.kernel_size,padding=padding_size)
        self.conv3 = nn.Conv2d(c,c,config.kernel_size,padding=padding_size)
        self.D_in_glimpse = g*g*c
        self.fc1 = nn.Linear(self.D_in_glimpse, h_g)

        # location layer
        D_in = 2
        self.fc2 = nn.Linear(D_in, h_g)

        # dropout
        self.drop = nn.Dropout(config.dropout)     

        # batchnorm layer        
        self.batchnorm_phi1 = nn.BatchNorm2d(c)
        self.batchnorm_phi2 = nn.BatchNorm2d(c)
        self.batchnorm_phi3 = nn.BatchNorm2d(c)
        self.batchnorm_phi4 = nn.BatchNorm1d(h_g)
        self.batchnorm_l = nn.BatchNorm1d(h_g)           

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        # phi = self.retina.foveate(x, l_t_prev) # get mulitiple patches from one glimpse
        phi_raw = self.retina.extract_patch(x, l_t_prev)

        phi = F.relu(self.batchnorm_phi1(self.conv1(phi_raw)))
        phi = F.relu(self.batchnorm_phi2(self.conv2(phi)))
        phi = F.relu(self.batchnorm_phi3(self.conv3(phi)))
        # change phi_out to shape [B,hidden_size]
        phi = phi.view(phi.shape[0],-1)
        # change dim by fc
        what = F.relu(self.drop(self.batchnorm_phi4(self.fc1(phi))))

        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        
        # increase dim by fc    
        where = F.relu(self.batchnorm_l(self.fc2(l_t_prev)))

        # feed to fc layer        
        g_t = F.relu6(what * where)
        
        return g_t

class context_network(nn.Module):
    """
    Uses the whole input image to produce the initial state for the 
    recurrent network to predict the first glimpse.

    Concretely, feeds the original image `input_img` through three
    convolution layers; then feeds the output through a fc layer to
    get the same size of hidden layer.   

    Args
    ----
    - num_channels: number of channels of dataset
    - kernel_size: the size of convolution kernel
    - hidden_size: the size of hidden layer
    - input_img: the input batch with size (N,C,H,W)

    Returns
    -------
    - h: output a vector for initial state of the top layer of rnn.(B,hidden_size)
    """
    def __init__(self, num_channels, kernel_size, hidden_size):
        super(context_network, self).__init__()       
        padding_size = int((kernel_size-1)*0.5);# make sure the output size is the same as input size. 
        self.conv1 = nn.Conv2d(num_channels,num_channels,kernel_size,padding=padding_size)
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size,padding=padding_size)
        self.conv3 = nn.Conv2d(num_channels,num_channels,kernel_size,padding=padding_size)
        self.fc = nn.Linear(num_channels*60*60, hidden_size)#60 is original image size, to do: make this auto indentify
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
       
    def forward(self, input_img):
        x = F.relu(self.bn1(self.conv1(input_img)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # output size: N,C,H,W
        # reshape the tensor as N x hidden_size to feed the fc layer        
        x = x.view(input_img.shape[0],-1)
        h = F.relu(self.fc(x))
        return h 


class core_network(nn.Module):
    """
    An RNN that maintains an internal state that integrates
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.

    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.

    In other words:

        `h_t = relu( fc(h_t_prev) + fc(g_t) )`

    Args
    ----
    - input_size: input size of the rnn.
    - hidden_size: hidden size of the rnn.
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    - h_t_prev1-2: a 2D tensor of shape (B, hidden_size). The
      hidden state vector for the previous timestep `t-1`.
    - cell_state_prev1-2: a 2D tensor of shape (B, hidden_size). The
      cell state vector for the previous timestep `t-1`.

    Returns
    -------
    - h_t1-2: a 2D tensor of shape (B, hidden_size). The hidden
      state vector for the current timestep `t`.
    - cell_state1-2: a 2D tensor of shape (B, hidden_size). The cell state
      vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size, config):
        super(core_network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn1 = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.rnn2 = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        
    def forward(self, g_t, h_t_prev1, h_t_prev2, cell_state_prev1, cell_state_prev2):
        h_t1,cell_state1 = self.rnn1(g_t,(h_t_prev1,cell_state_prev1))
        h_t2,cell_state2 = self.rnn2(h_t1,(h_t_prev2,cell_state_prev2))
        return h_t1, cell_state1, h_t2, cell_state2

class action_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the final output classification.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a softmax to create a vector of
    output probabilities over the possible classes.

    Hence, the environment action `a_t` is drawn from a
    distribution conditioned on an affine transformation
    of the hidden state vector `h_t`, or in other words,
    the action network is simply a linear softmax classifier.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - a_t: output probability vector over the classes.
    """
    def __init__(self, input_size, output_size):
        super(action_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t), dim=1)
        return a_t


class location_network(nn.Module):
    """
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args
    ----
    - input_size: input size of the fc layer, which is the hidden size
    - output_size: output size of the fc layer, which is 2: location (x,y)
    - std: standard deviation of the normal distribution.
    - h_t: the hidden state vector of the core network for
      the current time step `t`.

    Returns
    -------
    - mu: a 2D vector of shape (B, 2).
    - l_t: a 2D vector of shape (B, 2).
    """
    def __init__(self, input_size, output_size, config):
        super(location_network, self).__init__()
        self.std = config.std
        self.gamma = config.gamma # for stable distribution (temporally use Cauchy)
        self.fc1 = nn.Linear(input_size, int(output_size/2))
        self.fc2 = nn.Linear(int(output_size/2), output_size)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, h_t, l_t_prev=None):
        # compute mean
        mu = F.relu(self.drop(self.fc1(h_t.detach())))

        mu = torch.tanh(self.fc2(mu))
        
        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)
        l_t = mu + noise

        #noise.data.cauchy_(sigma=self.gamma)
        #l_t = mu + noise + l_t_prev # previous: l_t = mu + noise, now I change the mu + noise as the delta_x

        # bound between [-1, 1]
        l_t = torch.tanh(l_t)
        
        return mu, l_t

class combine_location_network(nn.Module):
    """
    A network that combines the Top-down and Bottom-up attention
    into a final location.

    - Top-down: get from hidden state, content based.
    - Bottom-up: get from saliency map driven Levy flight process with oscillation

    Concretely:

        `l_t = relu( fc( l_bu ) .* fc( l_td ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`. image patch;
      it equse the hidden size 
    - l_t_bu: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] gotten from saliency driven Levy flight process.
    - l_t_td: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] gotten from location network.

    Returns
    -------
    - l_out: a 2D vector of shape (B, 2).
    """
    def __init__(self, h_g, config):
        super(combine_location_network, self).__init__()
        
        # bottom-up location to hidden size
        D_in = 2
        self.fc1 = nn.Linear(D_in, int(h_g/2))
        # top-down location to hidden size
        self.fc2 = nn.Linear(D_in, int(h_g/2)) 

        # product of tow outputs of fc layers to location
        self.fc3 = nn.Linear(int(h_g/2),h_g)
        self.fc4 = nn.Linear(h_g, D_in)

        # dropout layer
        self.dropout_combined_att = nn.Dropout(config.dropout)        

        # batchnorm layer
        self.batchnorm_bu = nn.BatchNorm1d(int(h_g/2))
        self.batchnorm_td = nn.BatchNorm1d(int(h_g/2))
        self.batchnorm_c = nn.BatchNorm1d(h_g)         

    def forward(self, l_t_bu, l_t_td):
        # combine Top-down and Bottom-up attention        
        
        # increase dim by fc    
        l_bu_fc = F.relu(self.batchnorm_bu(self.fc1(l_t_bu)))               

        l_t_fc = F.relu(self.batchnorm_td(self.fc2(l_t_td)))

        # combine
        l_combine = F.relu(self.batchnorm_c(self.fc3(l_bu_fc * l_t_fc)))        
          
        l_out = torch.tanh(self.fc4(l_combine))
        
        return l_out

class baseline_network(nn.Module):
    """
    Regresses the baseline in the reward function
    to reduce the variance of the gradient update.

    Args
    ----
    - input_size: input size of the fc layer.
    - output_size: output size of the fc layer.
    - h_t: the hidden state vector of the core network
      for the current time step `t`.

    Returns
    -------
    - b_t: a 2D vector of shape (B, 1). The baseline
      for the current time step `t`.
    """
    def __init__(self, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = F.relu(self.fc(h_t.detach()))
        
        return b_t

# using WTA_IOR
class Levy_bottom_up_generator(object):
    """
    Generate the location gotten from saliency driven Levy flight process

    Args
    ----
    - batch_size: the number images in one batch.
    - image_size: a tuple contains (H, W) of a image
    - images: data with shape: B x C x H x W
    - alpha, beta, gamma, delta: stable distribution params
    - lold: the last location: B x 2
    - SM: Saliency Map: B x C x H x W

    Returns
    -------
    - lnew: the  location: B x 2

    """
    def __init__(self, batchsize, imagesize, config):        
        self.config = config
        
        self.SM_size = imagesize
        
        # range of finding local maximum
        self.left_half_patch_size = math.floor(config.patch_size/2)
        self.right_half_patch_size = self.left_half_patch_size + 1

    def __call__(self, SM_local_smooth):
        
        B = SM_local_smooth.shape[0]

        # candidate gotten from global maximum
        l_new_ind = SM_local_smooth.reshape(B,-1).argmax(1)
        # ind 2 sub
        l_new = torch.cat((l_new_ind.div(SM_local_smooth.shape[2]), l_new_ind.fmod(SM_local_smooth.shape[2])),dim=0).reshape(B,-1)
        #import pdb; pdb.set_trace()
        # inhibition of return
        for i in range(B):
            SM_local_smooth[i,0,(l_new[i,0]-self.left_half_patch_size) : (l_new[i,0]+self.right_half_patch_size), (l_new[i,1]-self.left_half_patch_size) : (l_new[i,1]+self.right_half_patch_size)] = 0       

        return l_new.float().div(self.SM_size[0] - 1).mul(2).sub(1), SM_local_smooth


# using the Hybrid Constrained Search algorithm, Boccignone & Ferraro, 2011, bad!!!
"""
class Levy_bottom_up_generator(object):
    -------------------
    Generate the location gotten from saliency driven Levy flight process

    Args
    ----
    - batch_size: the number images in one batch.
    - image_size: a tuple contains (H, W) of a image
    - images: data with shape: B x C x H x W
    - alpha, beta, gamma, delta: stable distribution params
    - lold: the last location: B x 2
    - SM: Saliency Map: B x C x H x W

    Returns
    -------
    - lnew: the  location: B x 2

    --------------
    def __init__(self, batchsize, imagesize, config):        
        self.config = config
        
        self.SM_size = imagesize
        # integration step
        self.dt = config.dt   
        # stable noise parameters
        self.alpha = config.alpha
        self.gamma = config.gamma        
        # oscillation parameters
        self.scale_oscillation = config.scale_oscillation
        self.period = config.period
        # direct comparison threshold
        self.threshold = config.threshold
        # Metropolis algorithm temperature
        self.T = config.T
        # range of finding local maximum
        self.kernel_size = math.floor(imagesize[0] / 12) * 2 + 1
        
        # counter of calling this class for the period term in langevin equation
        self.t = 0;

    def stblrnd(self, alpha, beta, gamma, delta, *argv):
        ---------
        STBLRND alpha-stable random number generator.
        draws a sample from the Levy 
        alpha-stable distribution with characteristic exponent ALPHA, 
        skewness BETA, scale parameter GAMMA and location parameter DELTA.
        ALPHA,BETA,GAMMA and DELTA must be scalars which fall in the following 
        ranges :
           0 < ALPHA <= 2
           -1 <= BETA <= 1  
           0 < GAMMA < inf 
           -inf < DELTA < inf


        R = STBLRND(ALPHA,BETA,GAMMA,DELTA,M,N,...) returns an M-by-N-by-... 
        array.   


        References:
        [1] J.M. Chambers, C.L. Mallows and B.W. Stuck (1976) 
            "A Method for Simulating Stable Random Variables"  
            JASA, Vol. 71, No. 354. pages 340-344  

        [2] Aleksander Weron and Rafal Weron (1995)
            "Computer Simulation of Levy alpha-Stable Variables and Processes" 
            Lec. Notes in Physics, 457, pages 379-392


        ---------
        
        # get output size
        sizeOut = []
        for arg in argv:
            sizeOut.append(arg)

        if self.config.use_gpu:
            size_bechmark = torch.zeros(sizeOut).cuda()
        else:
            size_bechmark = torch.zeros(sizeOut)

        # See if parameters reduce to a special case, if so be quick, if not 
        # perform general algorithm

        if alpha == 2: 
           # Gaussian distribution 
            r = torch.mul(torch.randn_like(size_bechmark), 2**0.5);

        elif int(alpha == 1) & int(beta == 0):   
            # Cauchy distribution
            r = torch.tan(torch.rand_like(size_bechmark).mul(2).sub(1).mul(math.pi/2)); 

        elif int(alpha == .5) & int(abs(beta) == 1):
            # Levy distribution (a.k.a. Pearson V)
            r = beta / torch.randn_like(size_bechmark).pow(2)

        elif beta == 0:              
            # Symmetric alpha-stable
            V = torch.rand_like(size_bechmark).mul(2).sub(1).mul(math.pi/2)
            W = -torch.log(torch.rand_like(size_bechmark));          
            r = torch.sin(V.mul(alpha)) / torch.cos(V).pow(1/alpha) * torch.cos(V.mul(1-alpha)).div(W).pow( (1-alpha)/alpha ) 

        elif alpha != 1:                
            # General case, alpha not 1
            V = torch.rand_like(size_bechmark).mul(2).sub(1).mul(math.pi/2)
            W = -torch.log(torch.rand_like(size_bechmark));        
            const = beta * math.tan(math.pi*alpha/2);
            B = math.atan( const );
            S = (1 + const * const)**(1/(2*alpha));        
            r = torch.sin(V).mul(alpha).add(B).mul(S) / torch.cos(V).pow(1/alpha) * (torch.cos(V.mul(1-alpha).sub(B)) / W).pow((1-alpha)/alpha)

        else:                             
            # General case, alpha = 1
            V = torch.rand_like(size_bechmark).mul(2).sub(1).mul(math.pi/2)
            W = -torch.log(torch.rand_like(size_bechmark));          
            piover2 = math.pi/2;
            sclshftV =  piover2 + V.mul(beta) ;           
            r = (torch.tan(V).mul(sclshftV) - torch.log(W.mul(piover2).mul(torch.cos(V)).div(sclshftV)).mul(beta)).mul(1/piover2)
                  
       
        # Scale and shift
        if alpha != 1:
           r = r.mul(gamma).add(delta);
        else:
           r = r.mul(gamma).add((2/math.pi) * beta * gamma * math.log(gamma) + delta)

        return r.squeeze()

    def jumping_length(self,SM,lold):
        # sampling jumping length from cauchy_levy_distribution
        # modifing phi

        beta = -1
        phi = torch.exp(beta * (SM[range(lold.shape[0]),0,lold[:,0],lold[:,1]].view(lold.shape[0],1,1,1) - SM))
        sum_normalize = torch.sum(torch.sum(phi,2,True),3,True)
        phi = phi / sum_normalize # shape[B,1,H,W]
        phi = phi.reshape(phi.shape[0],phi.shape[2]*phi.shape[3]) # shape[B,H*W]
        # all possible jumping length
        [x, y] = torch.meshgrid(torch.arange(SM.shape[2]), torch.arange(SM.shape[3]))
        x = x.unsqueeze(0)# unsqueeze for broadcast
        y = y.unsqueeze(0)
        length2 = (x - lold[range(lold.shape[0]),0].reshape(-1,1,1)).reshape(lold.shape[0],SM.shape[2],SM.shape[3])**2 + (y - lold[range(lold.shape[0]),1].reshape(-1,1,1)).reshape(lold.shape[0],SM.shape[2],SM.shape[3])**2 # shape [B,1,H,W]
        length2 = length2.reshape(length2.shape[0],length2.shape[1]*length2.shape[2]) # shape: [B, H*W]
        sorted_len2 = torch.unique(length2,dim=1)
        phi_sort = torch.zeros(sorted_len2.shape,dtype=torch.float)
        for i in range(sorted_len2.shape[0]):
            for j in range(sorted_len2.shape[1]):
                phi_sort[i,j] = torch.sum(phi[i,length2[i,:] == sorted_len2[i,j]])
        # distribution

        D = 0.8
        pdf = D * phi_sort / (D**2 + sorted_len2.float()) # shape [B,-1]
        # get cdf and normalize
        cdf = torch.cumsum(pdf,dim=1)
        cdf = cdf / cdf[range(cdf.shape[0]),-1].reshape(-1,1)
        # generate the random number
        #import pdb;pdb.set_trace()

        index = cdf.shape[1] - torch.sum(cdf > torch.rand(lold.shape[0],1),dim=1) # find the first non-zero element index       

        jumping_len = sorted_len2[range(sorted_len2.shape[0]),index]
        return torch.sqrt(jumping_len.float()) # shape [B,1]

    def langevin_random_search(self, lold, dSMx, dSMy):
        # lold, lnew: B x 2, [0,real_size]               
        # pure noise
        eta_x = self.stblrnd(self.alpha,0,self.gamma,0, lold.shape[0], 1); #Levy noise
        eta_y = self.stblrnd(self.alpha,0,self.gamma,0, lold.shape[0], 1); #Levy noise

        # follow the paper using weighted Cauchy-Levy distribution, MODELLING EYE-MOVEMENT CONTROL VIA A CONSTRAINED SEARCH APPROACH
        jumping_length = self.jumping_length(SM,lold)
        alpha = torch.rand_like(jumping_length) * 2 * math.pi
        eta_x = jumping_length * torch.cos(alpha)
        eta_y = jumping_length * torch.sin(alpha)
        
        # indexing partial derivations, in python, 1:3 is different with range(1,3)
        drift_x = dSMx[range(lold.shape[0]),0,lold[:,0],lold[:,1]]
        drift_y = dSMy[range(lold.shape[0]),0,lold[:,0],lold[:,1]]
        
        # numerical integration of langevin equation        
        lnew = torch.zeros_like(lold).long() 
                
        lnew[:,0] = lold[:,0] + (eta_x.add(self.scale_oscillation * math.cos(2*math.pi*self.t/self.period)) - drift_x).mul(self.dt).long()    # oscillation part can improve the performance?
        lnew[:,1] = lold[:,1] + (eta_y.add(self.scale_oscillation * math.cos(2*math.pi*self.t/self.period)) - drift_y).mul(self.dt).long() # oscillation part can improve the performance?

        # constrain
        lnew[lnew.abs().ge(self.SM_size[0]-1)] = self.SM_size[0] - 1
        lnew[lnew.le(0)] = 0

        return lnew

    def __call__(self, lold, SM, SM_local_smooth, dSMx, dSMy):
        self.t += self.dt
        # argmax(local_SM(r)) 
    
        # calculate coordinate for each batch samle (padding considered)
        from_x, from_y = lold[:, 0], lold[:, 1]
   
        B = lold.shape[0]

        if self.config.use_gpu:
            theta = torch.cuda.FloatTensor(B*2,3).fill_(0)            
        else:
            theta = torch.zeros(B*2,3)                    

        # see onenote of affine transform for this algorithm (Pytorch theta is different with cv2's affine matrix)
        theta[torch.arange(0,B*2,2),0] = self.kernel_size / self.SM_size[1] #/W
        theta[torch.arange(1,B*2,2),1] = self.kernel_size / self.SM_size[0] # / H
        theta[torch.arange(0,B*2,2),2] = from_x
        theta[torch.arange(1,B*2,2),2] = from_y
        theta = theta.reshape((B,2,3))

        grid = F.affine_grid(theta, torch.Size((B,1,self.kernel_size,self.kernel_size)))
        # focus of attention neighborhood
        FOA = F.grid_sample(SM,grid,mode='nearest',padding_mode='zeros')
        # pre allocate mem, all rejected cases will keep the old value, and transfer l from [-1, 1] to [0,real_size-1]
        l_new = lold.add(1).mul(self.SM_size[1]-1).div(2).long()
        lold = lold.add(1).mul(self.SM_size[1]-1).div(2).long()
        # candidate gotten from directly comparing
        l_compare_ind = FOA.reshape(B,-1).argmax(1)
        l_compare = torch.zeros_like(lold).long()
        # ind 2 sub
        l_compare[:,1] = l_compare_ind.fmod(self.kernel_size)
        l_compare[:,0] = l_compare_ind.div(self.kernel_size)

        # shift to the coordinate with the origin at left-bottom corner
        l_compare = l_compare - 5 + lold

        # Hybrid Constrained Search algorithm
        # direct comparison ind of acceptence
        accept_ind = SM_local_smooth[range(B),0,l_compare[:,0],l_compare[:,1]] - SM_local_smooth[range(B),0,lold[:,0],lold[:,1]] > self.threshold
        # store the acceptence of direct comparison
        l_new[accept_ind,:] = l_compare[accept_ind,:]

        if accept_ind.sum() < l_compare.shape[0]:
            # choose those need random search
            l_need_random_search = lold[~accept_ind,:]
            # random search by langenvin equation
            l_random_searched = self.langevin_random_search(l_need_random_search, dSMx, dSMy)
                
            # Metropolis algorithm        
            delta_SM = SM_local_smooth[~accept_ind,0,l_random_searched[:,0],l_random_searched[:,1]] - SM_local_smooth[~accept_ind,0,l_need_random_search[:,0],l_need_random_search[:,1]]         
            
            accept_ind_rl = delta_SM > 0
            # store
            # does not assign values: l_new[~accept_ind,:][accept_ind_rl,:] = l_random_searched[accept_ind_rl,:]
            tmp = l_new[~accept_ind,:]
            tmp[accept_ind_rl,:] = l_random_searched[accept_ind_rl,:]
            l_new[~accept_ind,:] = tmp
            # pure random        
            accept_ind_pr = torch.rand((~accept_ind_rl).sum(),device=accept_ind_rl.device) < torch.exp(delta_SM[~accept_ind_rl]/self.T)
            # store
            # does not assign values: l_new[~accept_ind,:][~accept_ind_rl,:][accept_ind_pr,:] = l_random_searched[~accept_ind_rl,:][accept_ind_pr,:]
            tmp1  = l_new[~accept_ind,:]
            tmp2 = tmp1[~accept_ind_rl,:]
            
            tmp3 = l_random_searched[~accept_ind_rl,:]
            
            tmp2[accept_ind_pr,:] = tmp3[accept_ind_pr,:]
            
            tmp1[~accept_ind_rl,:] = tmp2
            l_new[~accept_ind,:] = tmp1

        return l_new.float().div(self.SM_size[0] - 1).mul(2).sub(1)
"""



