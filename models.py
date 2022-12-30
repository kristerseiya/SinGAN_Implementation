import torch
from torch import nn
import torch.nn.functional as F
from math import ceil
import utils

class ConvBatchNormLeakyBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size=3,stride=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,output_channel,kernel_size,stride,padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x

# a generator for SinGAN
# takes two inputs (noise and generated image from previous generator)
#
# z ----
#       |
#       + ---- G -- + --- output
#       |           |
# lr ----------------
#
class AddSkipGenerator(nn.Module):
    # constructor inputs:
    # 1. channels:
    #   a list of number of channels for each convolutional layer
    #   len(channels) must be equal to # of convolutional layer + 1
    # 2. kernels:
    #   a list of kernel-sizes for each convolutional layer
    #   len(kernels) must be equal to # of convolutional layer
    #
    def __init__(self, channels, kernels, padding='gaussian'):
        assert(len(channels)==(len(kernels)+1))
        super().__init__()
        num_conv = len(channels) - 1
        num_pad = sum(kernels) - len(kernels)

        # input is padded in order to match the input and output size
        self.pad = utils.GaussianPad2d(num_pad//2, num_pad - num_pad//2,
                                       num_pad//2, num_pad - num_pad//2,
                                       scale=1.)
        self.convlist = nn.ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channels[0],
                                                         channels[1],
                                                         kernel_size=kernels[0]))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channels[i],
                                                         channels[i+1],
                                                         kernel_size=kernels[i]))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channels[-2],
                                           channels[-1],
                                           kernels[-1],1))

    def forward(self, z, lr):
        x = self.pad(z + lr)
        for l in self.convlist:
            x = l(x)
        return x + lr

# critic (used for WGAN/WGAN-GP) with only convolutional num_layers
#
class ConvCritic(nn.Module):
    # constructor inputs:
    # 1. input_size:
    #   a tuple or list of input tensor size
    # 2. channels:
    #   a list of number of channels for each convolutional layer
    #   len(channels) must be equal to # of convolutional layer + 1
    # 3. kernels:
    #   a list of kernel-sizes for each convolutional layer
    #   len(kernels) must be equal to # of convolutional layer
    # 4. strides
    #   a list of strides for each convolutional layer
    #   len(strides) must be equal to # of convolutional layer
    #   if no input is given, 1-stride is done for every convolutional layer
    # 5. padding
    #   if true, zero padding is done at the input to match the input size and output size
    #   (input size and output size will not match if strides are not 1s)
    #   by default, no padding is done (works better without padding)
    #
    def __init__(self, channels, kernels, strides=None):

        assert(len(channels) == (len(kernels)+1))
        if strides is None:
            strides = [1] * len(kernels)
        num_conv = len(channels) - 1

        super().__init__()
        self.convlist = nn.ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channels[0],
                                                         channels[1],
                                                         kernel_size=kernels[0],
                                                         stride=strides[0]))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channels[i],
                                                         channels[i+1],
                                                         kernel_size=kernels[i],
                                                         stride=strides[i]))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channels[-2],
                                           channels[-1],
                                           kernels[-1],
                                           strides[-1]))

    def forward(self, x):
        # x = self.pad(x)
        for l in self.convlist:
            x = l(x)
        return x

# class used to store the trained generators, reconstruct, and sample random images
#
class SinGAN(nn.Module):
    # constructor inputs:
    # 1. scale
    #   scale factor for each downsampling
    # 2. trained_size:
    #   image size of image trained for. optional
    # 3. G
    #   list of generators, optional
    # 4. z_amp
    #   list of standard deviation of noises for each generator
    # 5. Z
    #   list of fixed noises for each generator for reconstruction
    # 6. recimg
    #   list of reconstructed images for each generator
    #   if G, z_amp, Z are given and this isn't,
    #   it will automatically reconstruct the image for you
    #
    def __init__(self, z_base,
                 Gs=None, z_amps=None,
                 imgsizes=None, device=None):

        super().__init__()

        if Gs is None:
            Gs = nn.ModuleList()
        if z_amps is None:
            noise_levels = list()
        if imgsizes is None:
            imgsizes = list()

        if len(Gs) != len(z_amps):
            raise Exception("""G, noise_levels, imgsizes
                            must be lists with same length""")


        self.n_scale = len(Gs)

        if device is not None:
            for i in range(self.n_scale):
                Gs[i] = Gs[i].to(device)
                Zs[i] = Zs[i].to(device)

        self.z_base = z_base
        self.Gs = Gs
        self.z_amps = z_amps
        self.imgsizes = imgsizes
        self.device = device
        self.upfactors = list()
        self.recimgs = list()

        if type(self.Gs) == list:
            self.Gs = nn.ModuleList(self.Gs)

        if len(self.imgsizes) > 1:
            for prev, next in zip(self.imgsizes[:-1], self.imgsizes[1:]):
                self.upfactors.append(max(next[0] / float(prev[0]), next[1] / float(prev[1])))

        with torch.no_grad():
            for i in range(0, self.n_scale):
                if i == 0:
                    new_recimg = self.Gs[0](z_base, 0.)
                else:
                    prev = utils.upsample(self.recimgs[-1], self.upfactors[i-1])
                    new_recimg = self.Gs[i](0., prev)
                self.recimg.append(new_recimg)

    # append(self, netG, z_amp, fixed_z):
    #   appends a generator, noise information
    # 1. netG
    #   a new generator trained at one scale above
    #   it is recommended to disable gradient calculation of network by requires_grad = False
    # 2. z_amp
    #   standard deviation of noise input for the generator
    # 3. fixed_z
    #   fixed noise for the generator for reconstruction
    #
    def append(self, generator, noise_level, imgsize):
        generator.eval()
        utils.disable_grad(generator)
        self.Gs.append(generator)
        self.z_amps.append(noise_level)
        self.imgsizes.append(imgsize)
        self.n_scale = self.n_scale + 1
        if n_scale > 1:
            prev_size = self.imgsizes[-2]
            new_size = self.imgsizes[-1]
            self.upfactors.append(max(new_size[0] / float(prev_size[0])),
                                      new_size[1] / float(prev_size[1]))

        with torch.no_grad():
            if self.n_scale > 1:
                prev = utils.upsample(self.recimgs[-1], self.upfactors[-1])
                new_recimg = generator(0., prev)
            else:
                new_recimg = generator(self.z_base, 0.)

        self.recimgs.append(new_recimg)

        return self

    # reconstruct(self,scale_level=None)
    #   outputs a reconstructed image
    # 1. scale_level
    #   the scale level of reconstruction
    #   the smallest scale is 1, next scale up is 2, so forth
    #   if not given, it will output the reconstruction at the final scale
    #
    @torch.no_grad()
    def reconstruct(self, output_level=-1):
        if len(self.Gs) == 0:
            return None
        else:
            return self.recimgs[output_level]

    # sample(self,n_sample=1,scale_level=None)
    #   generates random samples
    # 1. n_sample
    #   a number of random samples
    # 2. scale_level
    #   the scale level of samples
    #
    @torch.no_grad()
    def sample(self, input_size=None,
               output_level=-1, n_sample=1):

        if len(self.Gs) == 0:
            return None
        if output_level < 0:
            output_level = self.n_scale + output_level

        if input_size is not None:
            z = self.z_amps[0] * torch.randn_like(n_sample, 3,
                                                  input_size[0],
                                                  input_size[1],
                                                  device=self.device)
        else:
            z = self.z_amps[0] * torch.randn_like(self.z_base)
        sample = self.Gs[0](z, 0.)
        for i in range(1, output_level+1):
            sample = utils.upsample(sample, self.upfactors[i-1])
            z = self.z_amps[i] * torch.randn_like(sample)
            sample = self.Gs[i](z,sample)
        return sample

    # inject(self,x,n_sample=1,insert_level=None,scale_level=None)
    # 1. n_sample
    #   a number of random samples
    # 2. insert_level
    #   specifies where to inject the image
    # 3. scale_level
    #   the scale level of the output image
    #
    @torch.no_grad()
    def inject(self, x, inject_level=-1, output_level=-1, n_sample=1):
        if self.n_scale == 0:
            return None
        if n_sample != 1:
            x = torch.cat(n_sample*[x],0)
        if inject_level < 0:
            inject_level = self.n_scale + inject_level
        if output_level < 0:
            output_level = self.n_scale + output_level

        z = self.z_amps[inject_level] * torch.randn_like(x)
        x = self.Gs[inject_level](z, x)
        for i in range(inject_level+1,output_level+1):
            x = utils.upsample(x, self.upfactors[i-1])
            z = self.z_amps[i] * torch.randn_like(x)
            x = self.Gs[i](z, x)
        return x

    def to(self, device):
        new_self = self.to(device)
        new_self.device = device
        for i in range(new_self.n_scale):
            # self.G[i] = self.G[i].to(device)
            new_self.recimgs[i] = new_self.recimgs[i].to(device)
        return new_self

    def walk(self, n, alpha, beta):
        z_t1 = 0.
        z_t = self.z_base

        if n > 0:
            xs = self.Gs[0](z_t, 0.)
            for j in range(1, self.n_scale):
                xs = utils.upsample(xs, self.upfactors[j-1])
                xs = self.Gs[j](0., xs)
        for i in range(1, n):
            z_diff = beta * (z_t - z_t1) + (1 - beta) * torch.rand_like(z_t)
            z_t1 = z_t
            z_t = alpha * self.z_base  + (1 - alpha) * (z_t + z_diff)
            x = self.Gs[0](z_t, 0.)
            for j in range(1, self.n_scale):
                x = utils.upsample(x, self.upfactors[j-1])
                x = self.Gs[j](0., x)
            xs = torch.cat([xs,x], dim=0)
        return xs


# loading and saving SinGAN

def save_singan(singan, path):
    torch.save({'n_scale': singan.n_scale,
                'scale': singan.upfactor,
                'trained_size': singan.imgsizes,
                'models': singan.Gs,
                'noise_amp': singan.z_amps,
                'fixed_noise': singan.Zs,
                'reconstructed_images': singan.recimgs
                }, path)
    return

def load_singan(path):
    load = torch.load(path)
    # singan = SinGAN(load['scale'], load['trained_size'], load['models'],
    #                 load['noise_amp'], load['fixed_noise'],load['reconstructed_images'])
    singan = SinGAN(load['fixed_noise'], Gs=load['models'], z_amps=load['noise_amp'], imgsizes=load['trained_size'])
    return singan
