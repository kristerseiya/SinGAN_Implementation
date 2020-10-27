import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .functions import *

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

class AIAOGenerator(nn.Module):
    def __init__(self,channel_config,kernel_size=3):
        super().__init__()
        num_conv = len(channel_config) - 1
        num_pad = (kernel_size // 2) * num_conv
        self.convlist = nn.ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[0],channel_config[1],kernel_size=kernel_size,padding=num_pad))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[i],channel_config[i+1],kernel_size=kernel_size))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channel_config[-2],channel_config[-1],kernel_size,1))

    def forward(self,z,lr):
        if z is None and lr is None:
            raise Exception("Both inputs cannot be Nonetype")
        if z is None:
            x = lr
        if lr is None:
            x = z
        # x = z + lr
        for l in self.convlist:
            x = l(x)
        return x + lr

class Critic(nn.Module):
    def __init__(self,channel_config,kernel_size=3):
        super().__init__()
        num_conv = len(channel_config) - 1
        num_pad = (kernel_size // 2) * num_conv
        self.convlist = nn.ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[0],channel_config[1],kernel_size=kernel_size,padding=num_pad))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[i],channel_config[i+1],kernel_size=kernel_size))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channel_config[-2],channel_config[-1],kernel_size,1))

    def forward(self,x):
        for l in self.convlist:
            x = l(x)
        return x

class SinGAN():
    def __init__(self, netG=None, imgsize=None, z_std=None, fixed_z=None):

        if netG is None:
            netG = []
        if imgsize is None:
            imgsize = []
        if z_std is None:
            z_std = []
        if fixed_z is None:
            fixed_z = []

        if len(fixed_z) != len(imgsize) or \
           len(fixed_z) != len(netG) or \
           len(fixed_z) != len(z_std):
            raise Exception("all list must have the same length")

        self.num_scales = len(netG)
        self.generators = netG
        self.z_std = z_std
        self.z0 = fixed_z
        self.imgsize = imgsize

    def append(self, netG, imgsize, z_std, fixed_z):
        self.generators.append(netG)
        self.imgsize.append(imgsize)
        self.z_std.append(z_std)
        self.z0.append(fixed_z)
        self.num_scales = self.num_scales + 1
        return

    def reconstruct(self,scale=None):
        if self.generators == []:
            return None
        if scale is None:
            scale = self.num_scales
        with torch.no_grad():
          zeros = torch.zeros_like(self.z0[0])
          rec = self.generators[0](self.z0[0],zeros)
          for i in range(1,scale):
              rec = F.interpolate(rec,self.imgsize[i])
              rec = self.generators[i](self.z0[i],rec)
        return rec

    def sample(self,num_sample=1,scale=None):
        if self.generators == []:
            return None
        if scale is None:
            scale = self.num_scales
        with torch.no_grad():
          zeros = torch.zeros_like(self.z0[0])
          zeros = torch.cat(num_sample*[zeros])
          z = self.z_std[0] * torch.randn_like(zeros)
          sample = self.generators[0](z,zeros)
          for i in range(1,scale):
              sample = F.interpolate(sample,self.imgsize[i])
              z = self.z_std[i] * torch.randn_like(sample)
              sample = self.generators[i](z,sample)
        return sample

    def inject(self,x,insert=None,scale=None):
        if insert is None:
            isert = self.num_scales
        if scale is None:
            scale = self.num_scales
        if (insert < 2) or (insert > self.num_scales):
            raise ValueError("insert argument must be 2 to %d" % self.num_scales-1)
        else:
            if self.generators == []:
                return None
            for i in range(insert-1,scale):
                x = F.interpolate(x,self.imgsize[i])
                z = self.z_std[i] * torch.randn_like(x)
                x = self.generators[i](z,x)
            return x

class SRSinGAN():
    def __init__(self, img, netG=None, imgsize=None, z_std=None, fixed_z=None):

        if netG is None:
            netG = []
        if imgsize is None:
            imgsize = []
        if z_std is None:
            z_std = []
        if fixed_z is None:
            fixed_z = []

        if len(fixed_z) != len(imgsize) or \
           len(fixed_z) != len(netG) or \
           len(fixed_z) != len(z_std):
            raise Exception("all list must have the same length")

        self.num_scales = len(netG) + 1
        self.lr = img
        self.generators = netG
        self.z_std = z_std
        self.z0 = fixed_z
        self.imgsize = imgsize

    def append(self, netG, imgsize, z_std, fixed_z):
        self.generators.append(netG)
        self.imgsize.append(imgsize)
        self.z_std.append(z_std)
        self.z0.append(fixed_z)
        self.num_scales = self.num_scales + 1
        return

    def reconstruct(self,scale=None):
        if self.generators == []:
            return self.lr
        if scale is None:
            scale = self.num_scales - 1
        with torch.no_grad():
          zeros = torch.zeros_like(self.z0[0])
          rec = self.generators[0](self.z0[0],self.lr)
          for i in range(1,scale):
              rec = F.interpolate(rec,self.imgsize[i])
              rec = self.generators[i](self.z0[i],rec)
        return rec

    def sample(self,num_sample=1,scale=None):
        if self.generators == []:
            return self.lr
        if scale is None:
            scale = self.num_scales - 1
        with torch.no_grad():
          zeros = torch.zeros_like(self.z0[0])
          zeros = torch.cat(num_sample*[zeros])
          z = self.z_std[0] * torch.randn_like(zeros)
          sample = self.generators[0](z,self.lr)
          for i in range(1,scale):
              sample = F.interpolate(sample,self.imgsize[i])
              z = self.z_std[i] * torch.randn_like(sample)
              sample = self.generators[i](z,sample)
        return sample

    def inject(self,x,insert=None,scale=None):
        if insert is None:
            insert = self.num_scales - 1
        if scale is None:
            scale = self.num_scales - 1
        if (insert < 2) or (insert > self.num_scales):
            raise ValueError("insert argument must be 2 to %d" % self.num_scales-1)
        else:
            if self.generators == []:
                return None
            for i in range(insert-1,scale):
                x = F.interpolate(x,self.imgsize[i])
                z = self.z_std[i] * torch.randn_like(x)
                x = self.generators[i](z,x)
            return x

def AdversarialLoss(disc_out):
    return -disc_out.mean()

def CriticLoss(disc_real,disc_fake):
    return -disc_real.mean() + disc_fake.mean()

def ReconstructionLoss(rec,target):
    criterion = nn.MSELoss()
    return criterion(rec,target)

def GradientPenaltyLoss(netD,real,fake):
    alpha = torch.rand(1,1).item()
    alpha = torch.full_like(real,alpha)
    interpolates = alpha * real + (1-alpha) * fake
    interpolates = torch.autograd.Variable(interpolates,requires_grad=True)
    Dout_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=Dout_interpolates, inputs=interpolates, \
                                  grad_outputs=torch.ones_like(Dout_interpolates), \
                                  create_graph=True,retain_graph=True,only_inputs=True)[0]
    D_grad_penalty = ((gradients.norm(2,dim=1)-1)**2).mean()

    return D_grad_penalty

def TrainSinGANOneScale(img,netG,netG_optim,netG_lrscheduler, \
                        netD,netD_optim,netD_lrscheduler, \
                        netG_chain,num_epoch, \
                        use_zero=True,batch_size=1, \
                        recloss_scale=10,gp_scale=0.1,z_std_scale=0.1, \
                        netG_iter=3,netD_iter=3,freq=0):

    imgsize = (img.size(-2),img.size(-1))
    zeros = torch.zeros_like(img)
    if batch_size != 1:
        batch = torch.cat(batch_size*[img])
        batch_zeros = torch.cat(batch_size*[zeros])
    else:
        batch = img
        batch_zeros = zeros

    if (netG_chain.num_scales == 0):
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

    netG_loss = []
    netD_loss = []
    wasserstein_distance = []

    netG.train()
    netD.train()

    for epoch in range(1,num_epoch+1):

        for i in range(netD_iter):
            netD_optim.zero_grad()

            # generate image
            if (first):
                z = z_std * torch.randn_like(batch_zeros)
                Gout = netG(z,batch_zeros)
            else:
                z = z_std * torch.randn_like(batch_zeros)
                base = netG_chain.sample(batch_size)
                base = F.interpolate(base,imgsize)
                Gout = netG(z,base)

            # train critic
            Dout_real = netD(batch)
            Dout_fake = netD(Gout.detach())
            D_loss = CriticLoss(Dout_real,Dout_fake)
            D_loss.backward()
            D_grad_penalty = GradientPenaltyLoss(netD,batch,Gout.detach()) * gp_scale
            D_grad_penalty.backward()
            D_loss_total = D_loss.item() + D_grad_penalty.item()
            netD_optim.step()
            netD_loss.append(D_loss_total)
            wasserstein_distance.append( Dout_real.item() - Dout_fake.item() )


        netD_lrscheduler.step()

        disableGrad(netD)

        for i in range(netG_iter):
            if (i!=0):
                if (first):
                  Gout = netG(z,batch_zeros)
                else:
                  base = netG_chain.sample(batch_size)
                  base = F.interpolate(base,imgsize)
                  Gout = netG(z,base)

            netG_optim.zero_grad()
            # train generator
            Dout_fake = netD(Gout)
            adv_loss = AdversarialLoss(Dout_fake)
            adv_loss.backward()
            if (first):
              rec = netG(fixed_z,zeros)
            else:
              rec = netG(fixed_z,prev_rec)
            rec_loss = ReconstructionLoss(rec,img) * recloss_scale
            rec_loss.backward()
            G_loss_total = adv_loss.item() + rec_loss.item()
            netG_optim.step()
            netG_loss.append(G_loss_total)

        netG_lrscheduler.step()

        enableGrad(netD)

        if (freq != 0) and (epoch % freq == 0):
            # show mean
            G_loss_mean = sum(netG_loss[-10:]) / 10
            D_loss_mean = sum(netD_loss[-10:]) / 10
            wasserstein_distance_mean = sum(wasserstein_distance[-10:]) / 10
            print("   generator loss   : {}".format(G_loss_mean))
            print("    critic loss     : {}".format(D_loss_mean))
            print("wasserstein_distance: {}".format(wasserstein_distance_mean))

            netG.eval()
            with torch.no_grad():
              # display sample from generator
              if (first):
                  z = z_std * torch.randn_like(zeros)
                  sample = netG(z,zeros)
                  rec = netG(fixed_z,zeros)
              else:
                  z = z_std * torch.randn_like(zeros)
                  base = netG_chain.sample()
                  base = F.interpolate(base,imgsize)
                  sample = netG(z,base)
                  rec = netG(fixed_z,prev_rec)

            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            showTensorImage(sample)
            plt.title("Random Sample")
            plt.subplot(1,2,2)
            showTensorImage(rec)
            plt.title("Reconstruction")
            plt.show()
            netG.train()

    return z_std, fixed_z, (netG_loss, netD_loss)
