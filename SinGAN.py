import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .functions import *
from math import ceil

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
    # 4. z_std
    #   list of standard deviation of noises for each generator
    # 5. Z
    #   list of fixed noises for each generator for reconstruction
    # 6. recimg
    #   list of reconstructed images for each generator
    #   if G, z_std, Z are given and this isn't,
    #   it will automatically reconstruct the image for you
    #
    def __init__(self, scale, trained_size=None, G=None, z_std=None, Z=None, recimg=None):

        if G is None:
            G = []
        if z_std is None:
            z_std = []
        if Z is None:
            Z = []
        if recimg is None:
            recimg = []

        if len(Z) != len(G) or \
           len(Z) != len(z_std):
            raise Exception("G, imgsize, z_std, Z must be lists with same length")

        self.n_scale = len(G)
        self.scale = scale
        self.trained_size = trained_size
        self.G = G
        self.z_std = z_std
        self.Z = Z
        self.recimg = recimg

        with torch.no_grad():
            for i in range(len(recimg),self.n_scale):
                if i == 0:
                    zeros = torch.randn_like(Z[0])
                    new_recimg = self.G[0](Z[0],zeros)
                else:
                    prev = F.interpolate(self.recimg[-1],scale_factor=1./self.scale)
                    new_recimg = self.G[i](Z[i],prev)
                self.recimg.append(new_recimg.detach())

    # append(self, netG, z_std, fixed_z):
    #   appends a generator, noise information
    # 1. netG
    #   a new generator trained at one scale above
    #   it is recommended to disable gradient calculation of network by requires_grad = False
    # 2. z_std
    #   standard deviation of noise input for the generator
    # 3. fixed_z
    #   fixed noise for the generator for reconstruction
    #
    def append(self, netG, z_std, fixed_z):
        self.G.append(netG)
        # self.imgsize.append(imgsize)
        self.z_std.append(z_std)
        self.Z.append(fixed_z.detach())

        with torch.no_grad():
            if self.n_scale > 0:
                prev = F.interpolate(self.recimg[-1],scale_factor=1./self.scale)
                new_recimg = netG(fixed_z,prev)
            else:
                zeros = torch.zeros_like(fixed_z)
                new_recimg = netG(fixed_z,zeros)

        self.recimg.append(new_recimg.detach())
        self.n_scale = self.n_scale + 1

        return

    # reconstruct(self,scale_level=None)
    #   outputs a reconstructed image
    # 1. scale_level
    #   the scale level of reconstruction
    #   the smallest scale is 1, next scale up is 2, so forth
    #   if not given, it will output the reconstruction at the final scale
    #
    @torch.no_grad()
    def reconstruct(self,scale_level=None):
        if scale_level is None:
            return self.recimg[-1]
        else:
            return self.recimg[scale_level-1]

    # sample(self,n_sample=1,scale_level=None)
    #   generates random samples
    # 1. n_sample
    #   a number of random samples
    # 2. scale_level
    #   the scale level of samples
    #
    @torch.no_grad()
    def sample(self,n_sample=1,scale_level=None):
        if self.G == []:
            return None
        if scale_level is None:
            scale_level = self.n_scale

        # with torch.no_grad():
        zeros = torch.zeros_like(self.Z[0])
        if n_sample != 1:
            zeros = torch.cat(n_sample*[zeros])
        z = self.z_std[0] * torch.randn_like(zeros)
        sample = self.G[0](z,zeros)
        for i in range(1,scale_level):
          sample = F.interpolate(sample,scale_factor=1./self.scale)
          z = self.z_std[i] * torch.randn_like(sample)
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
    def inject(self,x,n_sample=1,insert_level=None,scale_level=None):
        if self.G == []:
            return None
        if insert_level is None:
            insert_level = self.n_scale
        if scale_level is None:
            scale_level = self.n_scale
        if n_sample != 1:
            x = torch.cat(n_sample*[x],0)

        for _ in range(scale_level-insert_level+1):
            x = F.interpolate(x,(ceil(x.size(-2)/self.scale),ceil(x.size(-1)/self.scale)))
        for i in range(insert_level-1,scale_level):
            x = F.interpolate(x,scale_factor=1./self.scale)
            z = self.z_std[i] * torch.randn_like(x)
            x = self.G[i](z,x)
        return x


# loading and saving SinGAN

def saveSinGAN(singan,path):
    torch.save({'n_scale': singan.n_scale, \
                'scale': singan.scale, \
                'trained_size': singan.trained_size, \
                'models': singan.G, \
                'noise_amp': singan.z_std, \
                'fixed_noise': singan.Z, \
                'reconstructed_images': singan.recimg \
                },path)
    return

def loadSinGAN(path):
    load = torch.load(path)
    singan = SinGAN(load['scale'],load['trained_size'],load['models'], \
                    load['noise_amp'], load['fixed_noise'],load['reconstructed_images'])
    return singan

# calculates gradient penalty loss for WGAN-GP critic
def GradientPenaltyLoss(netD,real,fake):
    real = real.expand(fake.size())
    alpha = torch.rand(fake.size(0),1,1,1.device=fake.device)
    alpha = alpha.expand(fake.size())
    # alpha = torch.full_like(fake,alpha)
    interpolates = alpha * real + (1-alpha) * fake
    interpolates = torch.autograd.Variable(interpolates,requires_grad=True)
    Dout_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=Dout_interpolates, inputs=interpolates, \
                                  grad_outputs=torch.ones_like(Dout_interpolates), \
                                  create_graph=True,retain_graph=True,only_inputs=True)[0]

    grad_penalty = ((gradients.norm(2,dim=1)-1)**2).mean()

    return grad_penalty

# trains GAN for one scale of SinGAN
# 1. img
#   target image
# 2. netG
#   generator to train
# 3. netG_optim
#   optimizer for generator
# 4. netD
#   discriminator or critic
# 5. netD_optim
#   optimizer for discriminator or critic
# 6. netG_chain
#   SinGAN object that includes generator at lower scale
# 7. num_epoch
#   number of iteration
# 8. mode
#   specifies type of GAN
#   option: 'wgangp', 'wgan', 'gan'
#   gan type is 'wgangp' by default
# 9. netG_lrscheduler
#   learning rate scheduler for the generator, optional
# 10. netD_lrscheduler
#   learning rate scheduler for the critic, optional
# 11. use_zero
#   if True, use zero tensor for reconstruction
#   however, zero will not be used when training GAN at the lowest scale
# 12. batch_size
#   a number of samples generated for each parameter update
# 13. recloss
#   a function used for calculation of reconstruction loss, nn.MSELoss is used by default
# 14. recloss_scale
#   scalar multiplier for reconstruction loss, 10 is used by default
# 15. gp_scale
#   scalar multiplier for gradient penalty loss, 0.1 is used by default
# 16. clip_range
#   clipping range for critic's parameter when training for WGAN (no clipping is done for WGAN-GP)
#   0.01 by default
# 17. z_std_scale
#   scalar multiplier for input noise's standard deviation
# 18. netG_iter
#   a number of consecutive parameter updates for generator, 3 by default
# 19. netD_iter
#   a number of consecutive parameter updates for critic, 3 by default
# 20. freq
#   frequency of plotting random samples
# 21. figsize
#   size of the plot for generated samples
#
def TrainSinGANOneScale(img, \
                        netG,netG_optim, \
                        netD,netD_optim, \
                        netG_chain,num_epoch, \
                        mode='wgangp', \
                        netG_lrscheduler=None, netD_lrscheduler=None, \
                        use_zero=True,batch_size=1, \
                        recloss_fun=None,recloss_scale=10,gp_scale=0.1,clip_range=0.01, \
                        z_std_scale=0.1, \
                        netG_iter=3,netD_iter=3, \
                        freq=0, figsize=(15,15)):

    imgsize = (img.size(-2),img.size(-1))
    zeros = torch.zeros_like(img)

    if batch_size != 1:
        # batch = torch.cat(batch_size*[img])
        batch_zeros = torch.cat(batch_size*[zeros])
    else:
        # batch = img
        batch_zeros = zeros

    if (netG_chain.n_scale == 0):
        first = True
        z_std = z_std_scale
        fixed_z = z_std * torch.randn_like(img)
    else:
        first = False
        prev_rec = netG_chain.reconstruct()
        prev_rec = F.interpolate(prev_rec,imgsize)
        z_std = z_std_scale * torch.sqrt(F.mse_loss(prev_rec,img)).item()
        if use_zero:
            fixed_z = zeros
        else:
            fixed_z = z_std * torch.randn_like(img)

    if recloss==None:
        ReconstructionLoss = nn.MSELoss()
    else:
        ReconstructionLoss = recloss_fun

    netD_losses = []
    wasserstein_distances = []
    netG_losses = []
    rec_losses = []

    netG.train()
    netD.train()

    for epoch in range(1,num_epoch+1):

        enableGrad(netD)
        disableGrad(netG)

        for _ in range(netD_iter):

            # generate image
            if (first):
                z = z_std * torch.randn_like(batch_zeros)
                Gout = netG(z,batch_zeros)
            else:
                z = z_std * torch.randn_like(batch_zeros)
                base = netG_chain.sample(n_sample=batch_size)
                base = F.interpolate(base,imgsize)
                Gout = netG(z,base)

            # train critic
            netD_optim.zero_grad()

            Dout_real = netD(img)
            Dout_fake = netD(Gout.detach())
            if mode == 'gan':
                # calculate
                real_loss = F.binary_cross_entropy(Dout_real,torch.ones_like(Dout_real))
                real_loss.backward()
                fake_loss = F.binary_cross_entropy(Dout_fake,torch.zeros_like(Dout_fake))
                fake_loss.backward()
                D_loss_total = real_loss.item() + fake_loss.item()
                wdistance = 0.0
            elif mode == 'wgan':
                # calculate wasserstein distance
                wdistance_loss = Dout_fake.mean() - Dout_real.mean()
                wdistance_loss.backward()
                D_loss_total = wdistance_loss.item()
                wdistance = wdistance_loss.item()
            elif mode == 'wgangp':
                # calculate wasserstein distance and gradient penalty
                wdistance_loss = Dout_fake.mean() - Dout_real.mean()
                wdistance_loss.backward()
                grad_penalty_loss = GradientPenaltyLoss(netD,img,Gout) * gp_scale
                grad_penalty_loss.backward()
                D_loss_total = wdistance_loss.item() + grad_penalty_loss.item()
                wdistance = wdistance_loss.item()

            netD_optim.step()
            netD_losses.append( D_loss_total )
            wasserstein_distances.append( wdistance )

            if mode == 'wgan':
                for p in netD.parameters():
                    p.data.clamp_(-clip_range,clip_range)


        disableGrad(netD)
        enableGrad(netG)

        for _ in range(netG_iter):
            if (first):
                Gout = netG(z,batch_zeros)
            else:
                base = netG_chain.sample(n_sample=batch_size)
                base = F.interpolate(base,imgsize)
                Gout = netG(z,base)

            netG_optim.zero_grad()
            # train generator
            Dout_fake = netD(Gout)
            if mode == 'gan':
                adv_loss = F.binary_cross_entropy(Dout_fake,torch.ones_like(Dout_fake))
                adv_loss.backward()
            elif mode == 'wgan' or mode == 'wgangp':
                adv_loss = - Dout_fake.mean()
                adv_loss.backward()
            if (first):
                rec = netG(fixed_z,zeros)
            else:
                rec = netG(fixed_z,prev_rec)
            rec_loss = ReconstructionLoss(rec,img) * recloss_scale
            rec_loss.backward()
            G_loss_total = adv_loss.item() + rec_loss.item()

            netG_optim.step()
            netG_losses.append(G_loss_total)
            rec_losses.append(rec_loss)

        if netD_lrscheduler is not None:
            netD_lrscheduler.step()
        if netG_lrscheduler is not None:
            netG_lrscheduler.step()

        if (freq != 0) and (epoch % freq == 0):
            # show mean
            G_loss_mean = sum(netG_loss[-10:]) / 10
            D_loss_mean = sum(netD_loss[-10:]) / 10
            rec_loss_mean = sum(rec_losses[-10:]) / 10
            print("   generator loss   : {}".format(G_loss_mean))
            print("reconstruction loss : {}".format(rec_loss_mean))
            if mode == 'gan':
                print(" discriminator loss : {}".format(D_loss_mean))
            elif mode == 'wgan' or mode == 'wgangp':
                wasserstein_distance_mean = sum(wasserstein_distances[-10:]) / 10
                print("    critic loss     : {}".format(D_loss_mean))
                print("wasserstein_distance: {}".format(wasserstein_distance_mean))

            netG.eval()
            with torch.no_grad():
                # display sample from generator
                if (first):
                    tmp = torch.cat(7*[zeros],0)
                    z = z_std * torch.randn_like(tmp)
                    sample = netG(z,tmp)
                    rec = netG(fixed_z,zeros)
                else:
                    tmp = torch.cat(7*[zeros],0)
                    z = z_std * torch.randn_like(tmp)
                    base = netG_chain.sample(n_sample=7)
                    base = F.interpolate(base,imgsize)
                    sample = netG(z,base)
                    rec = netG(fixed_z,prev_rec)

                sample = torch.cat([sample,rec],0)

            plt.figure(figsize=figsize)
            showTensorImage(sample,4)
            plt.show()
            netG.train()

    return z_std, fixed_z, (netG_loss, netD_loss, wasserstein_distances, rec_losses)
