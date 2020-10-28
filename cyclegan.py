import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size=3,stride=1, \
                 padding=1,padding_mode="zeros"):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,output_channel,kernel_size,stride, \
                              padding=padding,padding_mode=padding_mode, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        a = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x + a

class DownBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size=3,stride=1, \
                 padding=1,padding_mode="zeros"):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,output_channel,kernel_size,stride, \
                              padding=padding,padding_mode=padding_mode, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.lrelu = nn.LeakyReLU(0.2)
        self.maxpl = nn.MaxPool2d(2)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        x = self.maxpl(x)
        return x

class UpBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size=3,stride=1, \
                 padding=1,padding_mode="zeros"):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(input_channel,output_channel,kernel_size,stride, \
                              padding=padding,padding_mode=padding_mode, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x

# (x - 1) // 2
class ResidualGenerator(nn.Module):
    def __init__(self,channel_config,n_resblock):
        super().__init__()
        self.down = nn.ModuleList()
        self.res = nn.ModuleList()
        self.up = nn.ModuleList()

        n_downsample = (len(channel_config) - 1 - n_resblock) // 2
        for i in range(n_downsample):
            self.down.append(DownBlock(channel_config[i],channel_config[i+1],padding=1))

        for i in range(n_downsample,n_downsample+n_resblock):
            self.res.append(ResidualBlock(channel_config[i],channel_config[i+1]))

        for i in range(n_downsample+n_resblock,n_downsample*2+n_resblock-1):
            self.up.append(UpBlock(channel_config[i],channel_config[i+1],padding=1))

        self.up.append(nn.Upsample(scale_factor=2))
        self.up.append(nn.Conv2d(channel_config[-2],channel_config[-1],3,1,padding=1))

    def forward(self,z,lr):
        x = z + lr
        for d in self.down:
            x = d(x)
        for r in self.res:
            x = r(x)
        for u in self.up:
            x = u(x)
        return x + lr

class CycleGAN(nn.Module):
    def __init__(self,netG_A,netG_B):
        self.mode = 'A'
        self.G_A = netG_A
        self.G_B = netG_B

    def switch(self):
        if self.mode == 'A':
            self.mode = 'B'
        elif self.mode == 'B':
            self.mode = 'A'

    def forward(self,x):
        if self.mode == 'A'
            self.G_A(x)
        elif self.mode =='B':
            self.G_B(x)
        return x

def trainCycleGAN(imgA,imgB, \
                  netG_A,netG_A_optim,netG_A_lrscheduler, \
                  netG_B,netG_B_optim,netG_B_lrscheduler, \
                  netD_A,netD_A_optim,netD_A_lrscheduler, \
                  netD_B,netD_B_optim,netD_A_lrscheduler, \
                  n_epoch):


    netD_A_optim.zero_grad()
    netD_B_optim.zero_grad()

    fake_imgB = netG_A(imgA)
    rec_imgA = netG_B(fake_imgB)
    fake_imgA = netG_B(imgB)
    rec_imgB = netG_A(fake_imgA)
    
    disc_real_A = netD_A(imgA)
    disc_fake_A = netD_A(fake_imgA.detach())
    disc_A_loss = CriticLoss(disc_real_A,disc_fake_A)
    disc_A_loss.backward()
    disc_A_gp = GradientPenalty(netD_A,imgA,fake_imgA.detach())
    disc_A_gp.backward()
    netD_A_optim.step()

    disc_real_B = netD_B(imgB)
    disc_fake_B = netD_B(fake_imgB.detach())
    disc_B_loss = CriticLoss(disc_real_B,disc_fake_B)
    disc_B_loss.backward()
    disc_B_gp = GradientPenalty(netD_B,imgB,fake_imgB.detach())
    disc_B_gp.backward()
    netD_B_optim.step()

    disableGrad(netD_A)
    disableGrad(netD_B)

    netG_A_optim.zero_grad()
    netG_B_optim.zero_grad()

    disc_fake_A = netD_A(fake_imgA)
    gen_A_loss = AdversarialLoss(disc_fake_A)
    gen_A_loss.backward()
    rec_loss_B = F.l1_loss(rec_imgB,imgB)
    rec_loss_B.backward()

    disc_fake_B = netD_B(fake_imgB)
    gen_B_loss = AdversarialLoss(disc_fake_B)
    gen_B_loss.backward()
    rec_loss_A = F.l1_loss(rec_imgA,imgA)
    rec_loss_A.backward()


def TrainSRSinGANOneScale(img,netG,netG_optim,netG_lrscheduler, \
                        netD,netD_optim,netD_lrscheduler, \
                        prev,num_epoch, \
                        use_zero=True,batch_size=1, \
                        recloss_scale=10,gp_scale=0.1,z_std_scale=0.1, \
                        netG_iter=1,netD_iter=1,freq=0):

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
            # if (first):
            #     z = z_std * torch.randn_like(batch_zeros)
            #     Gout = netG(z,batch_zeros)
            # else:
            #     z = z_std * torch.randn_like(batch_zeros)
            #     base = netG_chain.sample(batch_size)
            #     base = F.interpolate(base,imgsize)
            #     Gout = netG(z,base)
            Gout = netG(z,batch_zeros)

            # train critic
            Dout_real = netD(batch)
            Dout_fake = netD(Gout.detach())
            D_loss = - WassersteinDistance(Dout_real,Dout_fake)
            D_loss.backward()
            D_grad_penalty = GradientPenaltyLoss(netD,batch,Gout.detach()) * gp_scale
            D_grad_penalty.backward()
            D_loss_total = D_loss.item() + D_grad_penalty.item()
            netD_optim.step()
            netD_loss.append(D_loss_total)
            wasserstein_distance.append( - D_loss.item() )


        netD_lrscheduler.step()

        disableGrad(netD)

        for i in range(netG_iter):
            # if (i!=0):
            #     if (first):
            #       Gout = netG(z,batch_zeros)
            #     else:
            #       base = netG_chain.sample(batch_size)
            #       base = F.interpolate(base,imgsize)
            #       Gout = netG(z,base)

            netG_optim.zero_grad()
            # train generator
            Dout_fake = netD(Gout)
            adv_loss = AdversarialLoss(Dout_fake)
            adv_loss.backward()
            # if (first):
            #   rec = netG(fixed_z,zeros)
            # else:
            #   rec = netG(fixed_z,prev_rec)
            # rec_loss = ReconstructionLoss(rec,img) * recloss_scale
            # rec_loss.backward()
            G_loss_total = adv_loss.item()# + rec_loss.item()
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
                rec = netG(img,zeros)

            plt.figure(figsize=(15,15))
            showTensorImage(rec)
            plt.title("Reconstruction")
            plt.show()
            netG.train()

    return z_std, fixed_z, (netG_loss, netD_loss)


    