import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self,num_conv_layers,input_channel,
                 min_num_channel,max_num_channel,
                 channel_increase_with_layer=True):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        convlist = []
        bnlist = []
        if channel_increase_with_layer:
          num_channel = min_num_channel
        else:
          num_channel = max_num_channel
        convlist.append(nn.Conv2d(input_channel,num_channel,(3,3),1,bias=False,padding=(num_conv_layers,num_conv_layers)))
        bnlist.append(nn.BatchNorm2d(num_channel))
        for _ in range(num_conv_layers-2):
            if channel_increase_with_layer:
              new_num_channel = min(num_channel*2, max_num_channel)
            else:
              new_num_channel = max(num_channel//2, min_num_channel)
            new_num_channel = max(num_channel//2, min_num_channel)
            convlist.append(nn.Conv2d(num_channel,new_num_channel,(3,3),1,bias=False))
            bnlist.append(nn.BatchNorm2d(new_num_channel))
            num_channel = new_num_channel
        convlist.append(nn.Conv2d(num_channel,3,(3,3),1))
        self.convlist = nn.ModuleList(convlist)
        self.bnlist = nn.ModuleList(bnlist)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,z,lr):
        x = z + lr
        for i in range(self.num_conv_layers-1):
            x = self.convlist[i](x)
            x = self.bnlist[i](x)
            x = self.lrelu(x)
        x = self.convlist[-1](x)
        return x + lr

class Critic(nn.Module):
    def __init__(self,num_conv_layers,min_num_channel,max_num_channel,
                 channel_increase_with_layer=True):
        super().__init__()
        self.num_conv_layers = num_conv_layers
        convlist = []
        bnlist = []
        if channel_increase_with_layer:
          num_channel = min_num_channel
        else:
          num_channel = max_num_channel
        convlist.append(nn.Conv2d(3,num_channel,(3,3),padding=(num_conv_layers,num_conv_layers)))
        bnlist.append(nn.BatchNorm2d(num_channel))
        for _ in range(num_conv_layers-2):
            if channel_increase_with_layer:
              new_num_channel = min(num_channel*2, max_num_channel)
            else:
              new_num_channel = max(num_channel//2, min_num_channel)
            convlist.append(nn.Conv2d(num_channel,new_num_channel,(3,3),1))
            bnlist.append(nn.BatchNorm2d(new_num_channel))
            num_channel = new_num_channel
        convlist.append(nn.Conv2d(num_channel,3,(3,3),1))
        self.convlist = nn.ModuleList(convlist)
        self.bnlist = nn.ModuleList(bnlist)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        for i in range(self.num_conv_layers-1):
          x = self.convlist[i](x)
          x = self.bnlist[i](x)
          x = self.lrelu(x)
        x = self.convlist[-1](x)
        return x

class GeneratorChain():
    def __init__(self,netG_list,fixed_z,z_std_list,imgsize_list):
        self.num_scales = len(netG_list)
        self.generators = netG_list
        self.z0 = fixed_z
        self.z_std_list = z_std_list
        self.imgsizes = imgsize_list

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


def trainOneScale(img,netG,netG_optim,netG_lrscheduler,
                  netD,netD_optim,netD_lrscheduler,
                  netG_chain,
                  num_epoch,batch_size=1,
                  mse_scale=10,gp_scale=0.1,freq,device):

    batch = torch.cat(batch_size*[img])
    imgsize = (img.size(-2),img.size(-1))

    if (netG_chain.num_scales == 0):
        first = True
        zeros = torch.zeros((1,1,img.size(-2),img.size(-1)),device=device)
        batch_zeros = torch.cat(batch_size*[zeros])
        fixed_z = netG_chain.z0
        z_std = 1.0
        # z_std = img.std().item()
    else:
        first = False
        zeros = torch.zeros((1,3,img.size(-2),img.size(-1)),device=device)
        batch_zeros = torch.cat(batch_size*[zeros])
        # zeros = torch.zeros((1,1,img.size(-2),img.size(-1)),device=cuda)
        # batch_zeros = torch.cat(batch_size*[zeros])
        prev_rec = netG_chain.reconstruct()
        prev_rec = F.interpolate(prev_rec,imgsize)
        z_std = torch.sqrt(F.mse_loss(prev_rec,img)).item()

    netG_loss = []
    netD_loss = []

    netG.train()
    netD.train()
    for epoch in range(1,num_epoch+1):

        # generate image
        if (first):
            z = z_std * torch.randn((batch_size,1,img.size(-2),img.size(-1)),device=device)
            Gout = netG(z,batch_zeros)
        else:
            # z = z_std * torch.randn_like(batch_zeros)
            z = z_std * torch.randn((batch_size,3,img.size(-2),img.size(-1)),device=device)
            #z = torch.cat(3*[z],dim=1)
            base = netG_chain.sample(batch_size)
            base = F.interpolate(base,imgsize)
            Gout = netG(z,base)

        #for _ in range(3):
        netD_optim.zero_grad()

        # train critic
        Dout_real = netD(img)
        D_loss_real = - Cout_real.mean()
        D_loss_real.backward()
        Dout_fake = netD(Gout.detach())
        D_loss_fake = Cout_fake.mean()
        D_loss_fake.backward()
        # calculate gradient penalty for critic
        alpha = torch.rand(1,1).item()
        alpha = torch.full_like(batch,alpha)
        interpolates = alpha * batch + (1-alpha) * Gout.detach()
        interpolates = torch.autograd.Variable(interpolates,requires_grad=True)
        Dout_interpolates = netD(interpolates)
        gradients = torch.autograd.grad(outputs=Dout_interpolates, inputs=interpolates,
                                      grad_outputs=torch.ones_like(Cout_interpolates),
                                      create_graph=True,retain_graph=True,only_inputs=True)[0]
        D_grad_penalty = ((gradients.norm(2,dim=1)-1)**2).mean() * gp_scale
        D_grad_penalty.backward()
        D_loss = D_loss_real.item() + D_loss_fake.item() + D_grad_penalty.item()
        netD_optim.step()
        netD_loss.append(D_loss)

        for p in netD.parameters():  # reset requires_grad
          p.requires_grad = False

        #for i in range(3):
          # if (i!=0):
          #   if (first):
          #     Gout = netG(z,batch_zeros)
          #   else:
          #     base = netG_chain.sample(batch_size)
          #     base = F.interpolate(base,imgsize)
          #     Gout = netG(z,base)

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
        netG_loss.append(G_loss)

        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True

        netG_lrscheduler.step()
        netD_lrscheduler.step()

        if epoch % freq == 0:
            # show mean
            G_loss_mean = sum(netG_loss[-10:]) / 10
            D_loss_mean = sum(netD_loss[-10:]) / 10
            print("generator loss: {}".format(G_loss_mean))
            print("critic loss:    {}".format(D_loss_mean))

            netG.eval()
            with torch.no_grad():
              # display sample from generator
              if (first):
                  z = z_std * torch.randn((1,1,img.size(-2),img.size(-1)),device=device)
                  sample = netG(z,zeros)
                  rec = netG(fixed_z,zeros)
              else:
                  # z = z_std * torch.randn_like(img)
                  z = z_std * torch.randn((1,3,img.size(-2),img.size(-1)),device=device)
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
