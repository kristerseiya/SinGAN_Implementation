import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from . import utils
import numpy as np

# calculates gradient penalty loss for WGAN-GP critic
def GradientPenaltyLoss(netD,real,fake):
	if fake.size(0) > real.size(0) and real.size(0) == 1:
		real = real.expand(fake.size())
	if real.size(0) > fake.size(0):
		fake = fake.expand(real.size())
    # real = real.expand(fake.size())
    alpha = torch.rand(fake.size(0),1,1,1,device=fake.device)
    alpha = alpha.expand(fake.size())
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
# 6. singan
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

def train_singan_onescale(img, \
                          netG,netG_optim, \
                          netD,netD_optim, \
                          singan,num_epoch, \
                          mode='wgangp', \
                          netG_lrscheduler=None, netD_lrscheduler=None, \
                          use_zero=True,batch_size=1, \
                          recloss_fun=None,recloss_scale=10,gp_scale=0.1,clip_range=0.01, \
                          z_std_scale=0.1, \
                          netG_iter=3,netD_iter=3, \
                          log_freq=0, plot_freq=0, figsize=(15,15)):

    imgsize = (img.size(-2),img.size(-1))

    if  singan.n_scale == 0:
        z_std = z_std_scale
        fixed_z = z_std * torch.randn_like(img)
    else:
        prev_rec = singan.reconstruct()
        prev_rec = F.interpolate(prev_rec,imgsize)
        z_std = z_std_scale * torch.sqrt(F.mse_loss(prev_rec,img)).item()
        if use_zero:
            fixed_z = 0.
        else:
            fixed_z = z_std * torch.randn_like(img)

    if recloss_fun is None:
        ReconstructionLoss = nn.MSELoss()
    else:
        ReconstructionLoss = recloss_fun

    meta_data = np.zeros((num_epoch,(netG_iter+netD_iter)*2))

    netG.train()
    netD.train()

    for epoch in range(1,num_epoch+1):

        utils.enable_grad(netD)
        utils.disable_grad(netG)

        for i in range(netD_iter):

            # generate image
            z = z_std * torch.randn(batch_size,img.size(1),img.size(2),img.size(3),device=img.device)
            if singan.n_scale == 0:
                Gout = netG(z,0.)
            else:
                base = singan.sample(n_sample=batch_size)
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
            meta_data[epoch-1,i] = D_loss_total
            meta_data[epoch-1,i+netD_iter] = wdistance

            if mode == 'wgan':
                for p in netD.parameters():
                    p.data.clamp_(-clip_range,clip_range)


        utils.disable_grad(netD)
        utils.enable_grad(netG)

        for j in range(netG_iter):

            # generate image
            if singan.n_scale == 0:
                Gout = netG(z,0.)
            else:
                base = singan.sample(n_sample=batch_size)
                base = F.interpolate(base,imgsize)
                Gout = netG(z,base)

            # train generator
            netG_optim.zero_grad()
            Dout_fake = netD(Gout)
            if mode == 'gan':
                adv_loss = F.binary_cross_entropy(Dout_fake,torch.ones_like(Dout_fake))
                adv_loss.backward()
            elif mode == 'wgan' or mode == 'wgangp':
                adv_loss = - Dout_fake.mean()
                adv_loss.backward()
            if singan.n_scale == 0:
                rec = netG(fixed_z,0.)
            else:
                rec = netG(fixed_z,prev_rec)
            rec_loss = ReconstructionLoss(rec,img) * recloss_scale
            rec_loss.backward()
            G_loss_total = adv_loss.item() + rec_loss.item()

            netG_optim.step()

            meta_data[epoch-1,netD_iter*2+j] = G_loss_total
            meta_data[epoch-1,netD_iter*2+netG_iter+j] = rec_loss

        if netD_lrscheduler is not None:
            netD_lrscheduler.step()
        if netG_lrscheduler is not None:
            netG_lrscheduler.step()

        if (log_freq != 0) and (epoch % log_freq == 0):
            print("[{:d}/{:d}]".format(epoch,num_epoch))
            print("   generator loss   : {:.3f}".format(G_loss_total))
            print(" reconstruction loss: {:.3f}".format(rec_loss))
            if mode == 'gan':
                print(" discriminator loss : {:.3f}".format(D_loss_total))
            elif mode == 'wgan' or mode == 'wgangp':
                print("     critic loss    : {:.3f}".format(D_loss_total))
                print("wasserstein distance: {:.3f}".format(wdistance))

        if (plot_freq != 0) and (epoch % plot_freq == 0):
            netG.eval()
            with torch.no_grad():
                # display sample from generator
                if singan.n_scale == 0:
                    z = z_std * torch.randn(7,img.size(1),img.size(2),img.size(3),device=img.device)
                    sample = netG(z,0.)
                    rec = netG(fixed_z,0.)
                else:
                    z = z_std * torch.randn(7,img.size(1),img.size(2),img.size(3),device=img.device)
                    base = singan.sample(n_sample=7)
                    base = F.interpolate(base,imgsize)
                    sample = netG(z,base)
                    rec = netG(fixed_z,prev_rec)

                sample = torch.cat([sample,rec],0)

            plt.figure(figsize=figsize)
            utils.show_tensor_image(sample,4)
            netG.train()

    return z_std, fixed_z, meta_data


def train(r, scaled_img_list, device=None):

    singan = models.SinGAN(r)

    n_scale = len(scaled_img_list)
    tensor_list = utils.convert_images2tensors(scaled_img_list)
    nc = 8


    for i in range(num_scales):

        img = tensor_list[i]
        netG = models.AddSkipGenerator([3,nc,nc,nc,3],[3,3,3,3])
        if device is not None:
            netG = netG.to(device)
        netG.apply(xavier_uniform_weight_init)
        netD = models.ConvCritic([3,nc,nc,nc,1],[3,3,3,3])
        if device is not None:
            netD = netD.to(device)
        netD.apply(xavier_uniform_weight_init)
        netG_optim = torch.optim.Adam(netG.parameters(),lr=5e-4,betas=(0.5,0.999))
        netD_optim = torch.optim.Adam(netD.parameters(),lr=5e-4,betas=(0.5,0.999))
        netG_lrscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=netG_optim,milestones=[1600],gamma=0.1)
        netD_lrscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=netD_optim,milestones=[1600],gamma=0.1)

        print("Training Scale {}".format(i+1))

        z_std, fixed_z, _ = train_singan_onescale(img, \
                                                  netG,netG_optim, \
                                                  netD,netD_optim, \
                                                  netG_chain,2000, \
                                                  mode='wgangp', \
                                                  netG_lrscheduler=netG_lrscheduler, netD_lrscheduler=netD_lrscheduler, \
                                                  use_zero=True,batch_size=1, \
                                                  recloss_fun=recloss_fun,recloss_scale=10,gp_scale=0.1,clip_range=0.01, \
                                                  z_std_scale=0.1, \
                                                  netG_iter=3,netD_iter=3, \
                                                  log_freq=50, plot_freq=0, figsize=(15,15))

        netG.eval()
        utils.disable_grad(netG)
        singan.append(netG,z_std,fixed_z)

        nc = (int) (num_channel / r)

    return singan
