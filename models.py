import torch
from torch import nn
import torch.nn.functional as F
from math import ceil
from . import utils

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
# z ----|
#       + ---- G -- + --- output
# lr ---|-----------|
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
    def __init__(self,channels,kernels):
        assert(len(channels)==(len(kernels)+1))
        super().__init__()
        num_conv = len(channels) - 1
        num_pad = sum(kernels) - len(kernels)

        # input is padded in order to match the input and output size
        self.pad = nn.ZeroPad2d((num_pad//2,num_pad-num_pad//2, \
                                 num_pad//2,num_pad-num_pad//2))
        self.convlist = nn.ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channels[0],channels[1],kernel_size=kernels[0]))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channels[i],channels[i+1],kernel_size=kernels[i]))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channels[-2],channels[-1],kernels[-1],1))

    def forward(self,z,lr):
        x = z + lr
        x = self.pad(x)
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
    def __init__(self,channels,kernels,strides=None,padding=False):

        assert(len(channels)==(len(kernels)+1))
        if strides is None:
            strides = [1] * len(kernels)
        num_conv = len(channels) - 1
        if padding is True:
            num_pad = sum(kernels) - len(kernels)
        else:
            num_pad = 0

        super().__init__()
        self.pad = nn.ZeroPad2d((num_pad//2,num_pad-num_pad//2, \
                                 num_pad//2,num_pad-num_pad//2))
        self.convlist = nn.ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channels[0],channels[1],kernel_size=kernels[0],stride=strides[0]))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channels[i],channels[i+1],kernel_size=kernels[i],stride=strides[i]))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channels[-2],channels[-1],kernels[-1],strides[-1]))

    def forward(self,x):
        x = self.pad(x)
        for l in self.convlist:
            x = l(x)
        return x

# discriminator for conventional GAN
#
class Discriminator(nn.Module):
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
    # 5. n_dense_layer
    #   a number of dense layer at the end of the network
    #
    def __init__(self,input_size,channels,kernels,strides,n_dense_layer=1):
        assert(len(channels)==(len(kernels)+1))
        super().__init__()
        num_conv = len(channels) - 1

        self.convlist = nn.ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channels[0],channels[1],kernel_size=kernels[0],stride=strides[0]))
        for i in range(1, num_conv):
            self.convlist.append(ConvBatchNormLeakyBlock(channels[i],channels[i+1],kernel_size=kernels[i],stride=strides[i]))

        x = torch.randn(input_size)
        for l in self.convlist:
            x = l(x)
        self.num = x.nelement()
        self.linlist = nn.ModuleList()
        num = self.num
        for _ in range(n_dense_layer-1):
          self.linlist.append(nn.Linear(num,num//10))
          self.linlist.append(nn.LeakyReLU(0.2))
          num = num // 10
        if n_dense_layer > 0:
            self.linlist.append(nn.Linear(num,1))
        self.sgmd = nn.Sigmoid()

    def forward(self,x):
        for l in self.convlist:
            x = l(x)
        x = x.view(-1,self.num)
        for l in self.linlist:
            x = l(x)
        x = self.sgmd(x)
        return x

# class used to store the trained generators, reconstruct, and sample random images
#
class SinGAN():
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
    def __init__(self, scale, imgsize=None, G=None, z_amp=None, Z=None, recimg=None, device=None):

        if imgsize is None:
            imgsize = []
        if G is None:
            G = []
        if z_amp is None:
            z_amp = []
        if Z is None:
            Z = []
        if recimg is None:
            recimg = []

        if len(Z) != len(G) or \
           len(Z) != len(z_amp):
            raise Exception("G, z_amp, Z must be lists with same length")


        self.n_scale = len(G)

        if device is not None:
            for i in range(self.n_scale):
                G[i] = G[i].to(device)
                Z[i] = Z[i].to(device)

        self.scale = scale
        self.G = G
        self.z_amp = z_amp
        self.Z = Z
        self.recimg = recimg
        self.imgsize = imgsize
        self.device = device

        with torch.no_grad():
            for i in range(len(recimg),self.n_scale):
                if i == 0:
                    new_recimg = self.G[0](Z[0],0.)
                else:
                    prev = utils.upsample(self.recimg[-1],1./self.scale)
                    new_recimg = self.G[i](Z[i],prev)
                self.recimg.append(new_recimg.detach())

        if len(self.imgsize) < len(self.recimg):
            self.imgsize = [(x.size(-2),x.size(-1)) for x in self.recimg]

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
    def append(self, netG, z_amp, fixed_z):
        self.G.append(netG)
        self.z_amp.append(z_amp)
        if type(fixed_z) == torch.Tensor:
            self.Z.append(fixed_z.detach())
        else:
            self.Z.append(fixed_z)

        with torch.no_grad():
            if self.n_scale > 0:
                prev = utils.upsample(self.recimg[-1],1./self.scale)
                new_recimg = netG(fixed_z,prev)
            else:
                new_recimg = netG(fixed_z,0.)

        self.recimg.append(new_recimg.detach())
        self.imgsize.append((new_recimg.size(-2),new_recimg.size(-1)))
        self.n_scale = self.n_scale + 1

        return self

    # reconstruct(self,scale_level=None)
    #   outputs a reconstructed image
    # 1. scale_level
    #   the scale level of reconstruction
    #   the smallest scale is 1, next scale up is 2, so forth
    #   if not given, it will output the reconstruction at the final scale
    #
    @torch.no_grad()
    def reconstruct(self,output_level=None):
        if self.G == []:
            return None
        if output_level is None:
            return self.recimg[-1]
        else:
            return self.recimg[output_level]

    # sample(self,n_sample=1,scale_level=None)
    #   generates random samples
    # 1. n_sample
    #   a number of random samples
    # 2. scale_level
    #   the scale level of samples
    #
    @torch.no_grad()
    def sample(self,input_size=None, output_level=-1, n_sample=1):
        if self.G == []:
            return None
        if output_level < 0:
            output_level = self.n_scale + output_level

        if input_size is not None:
            z = self.z_amp[0] * torch.randn(n_sample,self.Z[0].size(1),input_size[0],input_size[1],device=self.device)
        else:
            z = self.z_amp[0] * torch.randn(n_sample,self.Z[0].size(1),self.Z[0].size(2),self.Z[0].size(3),device=self.device)
        sample = self.G[0](z,0.)
        for i in range(1,output_level+1):
            sample = utils.upsample(sample, 1./self.scale)
            z = self.z_amp[i] * torch.randn_like(sample)
            sample = self.G[i](z,sample)
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
    def inject(self,x,inject_level=-1,output_level=-1,n_sample=1):
        if self.G == []:
            return None
        if n_sample != 1:
            x = torch.cat(n_sample*[x],0)
        if inject_level < 0:
            inject_level = self.n_scale + inject_level
        if output_level < 0:
            output_level = self.n_scale + output_level

        z = self.z_amp[inject_level] * torch.randn_like(x)
        x = self.G[inject_level](z,x)
        for i in range(inject_level+1,output_level+1):
            x = utils.upsample(x, 1./self.scale)
            z = self.z_amp[i] * torch.randn_like(x)
            x = self.G[i](z,x)
        return x

    def to(self,device):
        self.device = device
        for i in range(self.n_scale):
            self.G[i] = self.G[i].to(device)
            self.recimg[i] = self.recimg[i].to(device)
            if type(self.Z[i]) is torch.Tensor:
                self.Z[i] = self.Z[i].to(device)
        return self

    # @torch.no_grad()
    # def forward_(self, input1, input2=0., input_level=0, output_level=-1):
    #     if input_level < 0:
    #         input_level = self.n_scale + input_level
    #     if output_level < 0:
    #         output_level = self.n_scale + output_level
    #
    #     x = self.G[input_level](input1,input2)
    #     for i in range(input_level+1,output_level+1):
    #         sample = utils.upsample(sample, 1./self.scale)
    #         z = self.z_amp[i] * torch.randn_like(sample)
    #     sample = self.G[i](z,sample)


    def walk(self,n,alpha,beta):
        z_t1 = 0.
        z_t = self.Z[0]

        if n > 0:
            xs = self.G[0](z_t,0.)
            for j in range(1,self.n_scale):
                xs = utils.upsample(xs,1./self.scale)
                xs = self.G[j](0.,xs)
        for i in range(1,n):
            z_diff = beta * (z_t - z_t1) + (1 - beta) * torch.rand_like(z_t)
            z_t1 = z_t
            z_t = alpha * self.Z[0]  + (1 - alpha) * (z_t + z_diff)
            x = self.G[0](z_t,0.)
            for j in range(1,self.n_scale):
                x = utils.upsample(x,1./self.scale)
                x = self.G[j](0.,x)
            xs = torch.cat([xs,x],0)
        return xs



# loading and saving SinGAN

def save_singan(singan,path):
    torch.save({'n_scale': singan.n_scale, \
                'scale': singan.scale, \
                'trained_size': singan.imgsize, \
                'models': singan.G, \
                'noise_amp': singan.z_amp, \
                'fixed_noise': singan.Z, \
                'reconstructed_images': singan.recimg \
                },path)
    return

def load_singan(path):
    load = torch.load(path)
    singan = SinGAN(load['scale'], load['trained_size'], load['models'], \
                    load['noise_amp'], load['fixed_noise'],load['reconstructed_images'])
    return singan

# def move_singan(singan,device):
#     for i in range(singan.n_scale):
#         singan.G[i] = singan.G[i].to(device)
#         singan.Z[i] = singan.Z[i].to(device)
#         singan.recimg[i] = singan.recimg[i].to(device)
#     return
