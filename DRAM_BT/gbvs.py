# gbvs saliency map, uncheck
import numpy as np
import torch
import torch.nn.functional as F
class saliency_map(object):
    """
    produce the saliency map of input image batch by GBVS method
    """
    
    def __init__(self):
        gaborparams = {
            'stddev': 2,
            'elongation': 2,
            'filterSize': -1,
            'filterPeriod': np.pi
        }

        self.params = {
            'gaborparams': gaborparams,
            'sigma_frac_act': 0.15,
            'sigma_frac_norm': 0.06,
            'max_level': 4,
            'thetas': [0, 45, 90, 135]
        }

        self.gaborKernels = getGaborKernels(self, gaborparams, thetas):
        self.gaussian_kernel = 
        
    # markov chain
    def markov_chain_solve(mat, tolerance):
        w,h = mat.shape
        diff = 1
        v = np.divide(np.ones((w, 1), dtype=np.float32), w)
        oldv = v
        oldoldv = v

        while diff > tolerance :
            oldv = v
            oldoldv = oldv
            v = np.dot(mat,v)
            diff = np.linalg.norm(oldv - v, ord=2)
            s = sum(v)
            if s>=0 and s< np.inf:
                continue
            else:
                v = oldoldv
                break

        v = np.divide(v, sum(v))

        return v
    # 
    def getGaborKernel(self, angle, phase):
        gp = self.gaborparams
        major_sd = gp['stddev']
        minor_sd = major_sd * gp['elongation']
        max_sd = max(major_sd, minor_sd)

        sz = gp['filterSize']
        if sz == -1:
            sz = math.ceil(max_sd * math.sqrt(10))
        else:
            sz = math.floor(sz / 2)

        psi = np.pi / 180 * phase
        rtDeg = np.pi / 180 * angle

        omega = 2 * np.pi / gp['filterPeriod']
        co = math.cos(rtDeg)
        si = -math.sin(rtDeg)
        major_sigq = 2 * pow(major_sd, 2)
        minor_sigq = 2 * pow(minor_sd, 2)

        vec = range(-int(sz), int(sz) + 1)
        vlen = len(vec)
        vco = [i * co for i in vec]
        vsi = [i * si for i in vec]

        # major = np.matlib.repmat(np.asarray(vco).transpose(), 1, vlen) + np.matlib.repmat(vsi, vlen, 1)
        a = np.tile(np.asarray(vco).transpose(), (vlen, 1)).transpose()
        b = np.matlib.repmat(vsi, vlen, 1)
        major = a + b
        major2 = np.power(major, 2)

        # minor = np.matlib.repmat(np.asarray(vsi).transpose(), 1, vlen) - np.matlib.repmat(vco, vlen, 1)
        a = np.tile(np.asarray(vsi).transpose(), (vlen, 1)).transpose()
        b = np.matlib.repmat(vco, vlen, 1)
        minor = a + b
        minor2 = np.power(minor, 2)

        a = np.cos(omega * major + psi)
        b = np.exp(-major2 / major_sigq - minor2 / minor_sigq)
        # result = np.cos(omega * major + psi) * exp(-major2/major_sigq - minor2/minor_sigq)
        result = np.multiply(a, b)

        filter1 = np.subtract(result, np.mean(result.reshape(-1)))
        filter1 = np.divide(filter1, np.sqrt(np.sum(np.power(filter1.reshape(-1), 2))))
        return torch.tensor(filter1) if self.config.use_gpu else filter1


    def getGaborKernels(self, gaborparams, thetas):
        gaborKernels = {}
        for th in thetas:
            gaborKernels[th] = {}
            gaborKernels[th]['0'] = getGaborKernel(gaborparams, th, 0)
            gaborKernels[th]['90'] = getGaborKernel(gaborparams, th, 90)

        return gaborKernels

    def getPyramids(self, image, max_level):
        imagePyr = []
        for i in range(max_level):
            gaussian_blur = F.conv2d(SM, self.gaussian_kernel, padding = self.padding_size)
            imagePyr.append(gaussian_blur[:,:,0:gaussian_blur.shape[2]:2,0:gaussian_blur.shape[3]:2])
        return imagePyr

    # colorFeatureMaps
    def colorFeatureMaps_compute(image_batch, L):
        # image_batch: B x C x H x W (Channel order: RGB)
        # L: B x 1 x H x W

        #CBY Feature Map
        min_rg = torch.max(image_batch[:,0:1,...],2)
        b_min_rg = torch.abs(torch.image_batch[:,2,...] - min_rg abs(np.subtract(b, min_rg)))
        CBY = b_min_rg / L

        #CRG Feature Map
        r_g = torch.abs(image_batch[:,0,...] - image_batch[:,1,...])
        CRG = r_g / L

        featMaps = {}
        featMaps['CBY'] = CBY
        featMaps['CRG'] = CRG
        featMaps['L'] = L
        return featMaps

    # orientationFeatureMaps
    def orientationFeatureMaps_compute(self, L, gaborparams, thetas):
        # L = Intensity Map        

        # kernels = getGaborKernels(gaborparams, thetas)
        featMaps = []
        for th in self.thetas:
            kernel_0  = self.kernels[th]['0']
            kernel_90 = self.kernels[th]['90']
            o1 = F.conv2d(L, kernel_0, padding = self.gabor_padding_size)
            o2 = F.conv2d(L, kernel_90, padding = self.gabor_padding_size)            
            o = o1.abs() + o2.abs()
            featMaps.append(o)

        return featMaps

    def calculateFeatureMaps(r, g, b, L, params):
        colorMaps = colorFeatureMaps_compute(r, g, b, L)
        orientationMaps = orientationFeatureMaps_compute(L, params['gaborparams'] , params['thetas'])
        allFeatureMaps = {
            0: colorMaps['CBY'],
            1: colorMaps['CRG'],
            2: colorMaps['L'],
            3: orientationMaps
        }
        return allFeatureMaps


    #graphBasedActivation
    def graphBasedActivation_calculate(map, sigma):
        [distanceMat, _, _] = loadGraphDistanceMatrixFor28x32()
        denom = 2 * pow(sigma, 2)
        expr = -np.divide(distanceMat, denom)
        Fab = np.exp(expr)

        map_linear = np.ravel(map, order='F')  # column major
        state_transition_matrix = np.zeros_like(distanceMat, dtype=np.float32)

        # calculating STM : w = d*Fab
        for i in xrange(distanceMat.shape[0]):
            for j in xrange(distanceMat.shape[1]):
                state_transition_matrix[i][j] = Fab[i][j] * abs(map_linear[i] - map_linear[j])

        # normalising outgoing weights of each node to sum to 1, using scikit normalize
        norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

        # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
        # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
        eVec = markov_chain_solve(norm_STM, 0.0001)
        processed_reshaped = np.reshape(eVec, map.shape, order='F')

        return processed_reshaped


    def __call__(self, image_batch):
        params = self.params        
        

        if self.num_channel == 3:
            # L = Intensity Map            
            L = torch.max(image_batch, 2)

            pyr = getPyramids(image_batch, self.max_level)
            
            L_pyr = getPyramids(L, self.max_level)

            featMaps = {
                0: [],
                1: [],
                2: [],
                3: []
            }

        # calculating feature maps
        # loop through pyramids
        for i in range(0, len(L_pyr)):
            p_r = r_pyr[i]
            p_g = g_pyr[i]
            p_b = b_pyr[i]
            p_L = L_pyr[i]

            featMaps[] = calculateFeatureMaps(p_r, p_g, p_b, p_L, params)

        # we calculate feature maps and then resize
        # F.interpolate(maps,size=(60,60),scale_factor=,)

        # calculating activation maps

        activationMaps = []
        activation_sigma = params['sigma_frac_act']*60 # the shape of map
        # loop four features
        for i in range(0,4):
            for map in featMaps[i]:
                activationMaps.append(graphBasedActivation_calculate(map, activation_sigma))


        # normalizing activation maps

        normalisedActivationMaps = []
        normalisation_sigma = params['sigma_frac_norm']*np.mean([32, 28])

        for map in activationMaps:
            normalisedActivationMaps.append(graphBasedActivation.normalize(map, normalisation_sigma))


        # combine normalised maps

        mastermap = normalisedActivationMaps[0]
        for i in range(1, len(normalisedActivationMaps)):
            mastermap = np.add(normalisedActivationMaps[i], mastermap)


        # post process

        gray = cv2.normalize(mastermap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # blurred = cv2.GaussianBlur(gray,(4,4), 4)
        # gray2 = cv2.normalize(blurred, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mastermap_res = cv2.resize(gray, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

        return mastermap_res