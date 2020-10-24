import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functions import *

class ConvBatchNormLeakyBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel=(3,3),stride=1,padding=(0,0)):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,output_channel,kernel,stride,padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x

class SinGenerator(nn.Module):
    def __init__(self,channel_config):
        super().__init__()
        num_conv = len(channel_config) - 1
        self.convlist = ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[i],channel_config[i+1],padding=(5,5)))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[i],channel_config[i+1]))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channel_config[-2],channel_config[-1]))

    def forward(self,z,lr):
        x = z + lr
        for l in self.convlist:
            x = l(x)
        return x + lr

class SinCritic(nn.Module):
    def __init__(self,channel_config):
        super().__init__()
        num_conv = len(channel_config) - 1
        self.convlist = ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[i],channel_config[i+1],padding=(5,5)))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[i],channel_config[i+1]))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channel_config[-2],channel_config[-1]))

    def forward(self,x):
        for l in self.convlist:
            x = l(x)
        return x

class SinGeneratorChain():
    def __init__(self, fixed_z, init_z_std=0.1, imgsize_list=[], netG_list=[], z_std_list=[]):
        self.num_scales = len(netG_list)
        self.generators = netG_list
        self.init_z_std = init_z_std
        self.z_std_list = z_std_list
        self.z0 = fixed_z
        self.imgsizes = imgsize_list

    def append(self, netG, imgsize, z_std):
        self.generators.append(netG)
        self.imgsizes.append(imgsize)
        self.z_std_list.append(z_std)
        self.num_scales = self.num_scales + 1
        return

    def reconstruct(self):
        if self.generators == []:
            return
        with torch.no_grad():
          zeros = torch.zeros_like(self.z0)
          rec = self.generators[0](self.z0,zeros)
          for i in range(1,self.num_scales):
              rec = F.interpolate(rec,self.imgsizes[i])
              zeros = torch.zeros_like(rec)
              rec = self.generators[i](zeros,rec)
        return rec

    def sample(self,num_sample=1):
        if self.generators == []:
            return
        with torch.no_grad():
          zeros = torch.zeros_like(self.z0)
          zeros = torch.cat(num_sample*[zeros])
          z = self.z_std_list[0] * torch.randn_like(zeros)
          sample = self.generators[0](z,zeros)
          for i in range(1,self.num_scales):
              sample = F.interpolate(sample,self.imgsizes[i])
              z = self.z_std_list[i] * torch.randn_like(sample)
              sample = self.generators[i](z,sample)
        return sample

def GradientPenaltyLoss(netD,real,fake,gp_scale):
    alpha = torch.rand(1,1).item()
    alpha = torch.full_like(batch,alpha)
    interpolates = real * batch + (1-alpha) * fake
    interpolates = torch.autograd.Variable(interpolates,requires_grad=True)
    Dout_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=Dout_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(Dout_interpolates),
                                  create_graph=True,retain_graph=True,only_inputs=True)[0]
    D_grad_penalty = ((gradients.norm(2,dim=1)-1)**2).mean() * gp_scale

    return D_grad_penalty

def TrainSinGANOneScale(img,netG,netG_optim,netG_lrscheduler,
                        netD,netD_optim,netD_lrscheduler,
                        netG_chain,num_epoch,
                        freq=0,batch_size=1,
                        mse_scale=10,gp_scale=0.1,z_std_scale=0.1,
                        netG_iter=3,netD_iter=3):

    batch = torch.cat(batch_size*[img])
    imgsize = (img.size(-2),img.size(-1))

    if (netG_chain.num_scales == 0):
        first = True
        fixed_z = netG_chain.z0
        zeros = torch.zeros_like(netG_chain.z0)
        batch_zeros = torch.cat(batch_size*[zeros])
        z_std = z_std_scale
        # z_std = img.std().item()
    else:
        first = False
        zeros = torch.zeros_like(img)
        batch_zeros = torch.cat(batch_size*[zeros])
        # zeros = torch.zeros((1,1,img.size(-2),img.size(-1)),device=cuda)
        # batch_zeros = torch.cat(batch_size*[zeros])
        prev_rec = netG_chain.reconstruct()
        prev_rec = F.interpolate(prev_rec,imgsize)
        z_std = z_std_scale * torch.sqrt(F.mse_loss(prev_rec,img)).item()

    netG_loss = []
    netD_loss = []

    netG.train()
    netD.train()

    for epoch in range(1,num_epoch+1):

        for
        # generate image
        if (first):
            z = z_std * torch.randn_like(batch_zeros)
            Gout = netG(z,batch_zeros)
        else:
            # z = z_std * torch.randn_like(batch_zeros)
            z = z_std * torch.randn_like(batch_zeros)
            #z = torch.cat(3*[z],dim=1)
            base = netG_chain.sample(batch_size)
            base = F.interpolate(base,imgsize)
            Gout = netG(z,base)

        for i in range(netD_iter):
            netD_optim.zero_grad()

            # train critic
            Dout_real = netD(img)
            D_loss_real = - Dout_real.mean()
            D_loss_real.backward()
            Dout_fake = netD(Gout.detach())
            D_loss_fake = Dout_fake.mean()
            D_loss_fake.backward()
            D_grad_penalty = GradientPenaltyLoss(netD,img,Gout.detach(),gp_scale)
            D_grad_penalty.backward()
            D_loss = D_loss_real.item() + D_loss_fake.item() + D_grad_penalty.item()
            netD_optim.step()
            netD_lrscheduler.step()
            netD_loss.append(D_loss)

        # for p in netD.parameters():  # reset requires_grad
        #     p.requires_grad = False
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
            adv_loss = - Dout_fake.mean()
            adv_loss.backward()
            if (first):
              rec = netG(fixed_z,zeros)
            else:
              rec = netG(zeros,prev_rec)
            rec_loss = F.mse_loss(rec,img) * mse_scale
            rec_loss.backward()
            # G_loss = - Cout_fake.mean() + rec_loss
            # G_loss.backward()
            G_loss = adv_loss.item() + rec_loss.item()
            netG_optim.step()
            netG_lrscheduler.step()
            netG_loss.append(G_loss)

        # for p in netD.parameters():  # reset requires_grad
        #     p.requires_grad = True
        enableGrad(netD)

        if (freq != 0) and (epoch % freq == 0):
            # show mean
            G_loss_mean = sum(netG_loss[-10:]) / 10
            D_loss_mean = sum(netD_loss[-10:]) / 10
            print("generator loss: {}".format(G_loss_mean))
            print("critic loss:    {}".format(D_loss_mean))

            netG.eval()
            with torch.no_grad():
              # display sample from generator
              if (first):
                  z = z_std * torch.randn_like(zeros)
                  sample = netG(z,zeros)
                  rec = netG(fixed_z,zeros)
              else:
                  # z = z_std * torch.randn_like(img)
                  z = z_std * torch.randn_like(zeros)
                  #z = torch.cat(3*[z],dim=1)
                  base = netG_chain.sample()
                  base = F.interpolate(base,imgsize)
                  sample = netG(z,base)
                  rec = netG(zeros,prev_rec)

            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            showTensorImg(sample)
            plt.title("Random Sample")
            plt.subplot(1,2,2)
            showTensorImg(rec)
            plt.title("Reconstruction")
            plt.show()
            netG.train()

    return z_std, (netG_loss, netD_loss)
