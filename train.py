
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils
import numpy as np
import config


# calculates gradient penalty loss for WGAN-GP critic
def GradientPenaltyLoss(netD, real, fake):
	# if fake.size(0) > real.size(0) and real.size(0) == 1:
	# 	real = real.expand(fake.size())
	# if real.size(0) > fake.size(0):
	# 	fake = fake.expand(real.size())
    real = real.expand(fake.size())
    alpha = torch.rand(fake.size(0), 1, 1, 1, device=fake.device)
    alpha = alpha.expand(fake.size())
    interpolates = alpha * real + (1-alpha) * fake
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    Dout_interpolates = netD(interpolates)
    gradients = torch.autograd.grad(outputs=Dout_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(Dout_interpolates),
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

def train_singan_onescale(img,
                          netG, netG_optim,
                          netD, netD_optim,
                          num_epoch, singan=None,
                          netG_lrscheduler=None,
                          netD_lrscheduler=None,
                          noise_scale=0.1, recloss_coef=10.,
                          gradpenalty_coef=0.1, clip_range=0.01,
                          gen_iter=3, crt_iter=3,
                          log_freq=0, plot_freq=0):

    imgsize = (img.size(-2),img.size(-1))

    if  singan == None:
        z_std = noise_scale
        fixed_z = z_std * torch.randn((1, 1, *imgsize), device=img.device)
    else:
        prev_rec = singan.reconstruct()
        prev_rec = F.interpolate(prev_rec, imgsize)
        z_std = noise_scale * torch.sqrt(F.mse_loss(prev_rec, img)).item()
        fixed_z = 0.

    ReconstructionLoss = nn.MSELoss()

    train_meta = np.zeros((num_epoch,
                         (gen_iter+crt_iter)*2))

    netG.train()
    netD.train()

    for epoch in range(1, num_epoch+1):

        utils.enable_grad(netD)
        utils.disable_grad(netG)

        for i in range(crt_iter):

            # generate image
            z = noise_scale * torch.randn_like(fixed_z)
            if singan == None:
                Gout = netG(z, 0.)
            else:
                base = singan.sample(n_sample=1)
                base = F.interpolate(base, imgsize)
                Gout = netG(z, base)

            # train critic
            netD_optim.zero_grad()
            Dout_real = netD(img)
            Dout_fake = netD(Gout.detach())
            if mode == 'wgan':
                # calculate wasserstein distance
                wdistance_loss = Dout_fake.mean() - Dout_real.mean()
                wdistance_loss.backward()
                D_loss_total = wdistance_loss.item()
                wdistance = wdistance_loss.item()
            elif mode == 'wgangp':
                # calculate wasserstein distance and gradient penalty
                wdistance_loss = Dout_fake.mean() - Dout_real.mean()
                wdistance_loss.backward()
                grad_penalty_loss = GradientPenaltyLoss(netD, img, Gout) * gradpenalty_coef
                grad_penalty_loss.backward()
                D_loss_total = wdistance_loss.item() + grad_penalty_loss.item()
                wdistance = wdistance_loss.item()

            netD_optim.step()
            train_meta[epoch-1,i] = D_loss_total
            train_meta[epoch-1,i+netD_iter] = wdistance

            if mode == 'wgan':
                for p in netD.parameters():
                    p.data.clamp_(-clip_range, clip_range)

        utils.disable_grad(netD)
        utils.enable_grad(netG)

        for j in range(gen_iter):

            # generate image
            if singan == None:
                Gout = netG(z, 0.)
            else:
                base = singan.sample(n_sample=1)
                base = F.interpolate(base, imgsize)
                Gout = netG(z, base)

            # train generator
            netG_optim.zero_grad()
            Dout_fake = netD(Gout)

            adv_loss = - Dout_fake.mean()
            adv_loss.backward()

            if singan == None:
                rec = netG(fixed_z, 0.)
            else:
                rec = netG(fixed_z, prev_rec)
            rec_loss = ReconstructionLoss(rec,img) * recloss_coef
            rec_loss.backward()
            G_loss_total = adv_loss.item() + rec_loss.item()

            netG_optim.step()

            train_meta[epoch-1, crt_iter*2+j] = G_loss_total
            train_meta[epoch-1, crt_iter*2 + gen_iter+j] = rec_loss

        if netD_lrscheduler != None:
            netD_lrscheduler.step()
        if netG_lrscheduler != None:
            netG_lrscheduler.step()

        if (log_freq != 0) and (epoch % log_freq == 0):
            print("[{:d}/{:d}]".format(epoch,num_epoch))
            print("   generator loss   : {:.3f}".format(G_loss_total))
            print(" reconstruction loss: {:.3f}".format(rec_loss))
            print("     critic loss    : {:.3f}".format(D_loss_total))
            print("wasserstein distance: {:.3f}".format(wdistance))

        if (plot_freq != 0) and (epoch % plot_freq == 0):
            netG.eval()
            with torch.no_grad():
                # display sample from generator
                if singan == None:
                    z = z_std * torch.randn(n_sample, img.size(1),
                                            img.size(2),img.size(3),
                                            device=img.device)
                    sample = netG(z, 0.)
                    rec = netG(fixed_z, 0.)
                else:
                    z = z_std * torch.randn(n_sample,img.size(1),
                                            img.size(2), img.size(3),
                                            device=img.device)
                    base = singan.sample(n_sample=n_sample)
                    base = F.interpolate(base, imgsize)
                    sample = netG(z, base)
                    rec = netG(fixed_z, prev_rec)

                sample = torch.cat([sample, rec],0)

            plt.figure(figsize=(15, 15))
            utils.show_tensor_image(sample, 4)
            netG.train()

    if singan == None:
        singan = model.SinGAN(fixed_z)

    singan.append(netG, z_std, imgsize)

    return singan, train_meta


def train(img_pyramid, device=None):

    n_scale = len(img_pyramid)
    tensor_list = utils.convert_images2tensors(img_pyramid)
    nc = 12
    singan = None

    for i in range(n_scale):

        img = tensor_list[i]
        netG = models.AddSkipGenerator([3, nc, nc, nc, 3], [3, 3, 3, 3])
        if device is not None:
            netG = netG.to(device)
        netG.apply(xavier_uniform_weight_init)
        netD = models.ConvCritic([3, nc, nc, nc, 1], [3, 3, 3, 3])
        if device is not None:
            netD = netD.to(device)
        netD.apply(xavier_uniform_weight_init)
        netG_optim = torch.optim.Adam(netG.parameters(),lr=5e-4, betas=(0.5,0.999))
        netD_optim = torch.optim.Adam(netD.parameters(),lr=5e-4, betas=(0.5,0.999))
        netG_lrscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=netG_optim,milestones=[1600],gamma=0.1)
        netD_lrscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=netD_optim,milestones=[1600],gamma=0.1)

        print("Training Scale {}".format(i+1))

        singan, _ = train_singan_onescale(img, netG, netG_optim,
                                          netD, netD_optim, n_epoch,
                                          singan=singan,
                                          netG_lrscheduler=netG_lrscheduler,
                                          netD_lrscheduler=netD_lrscheduler)

        nc = int(nc * 1.5)

    return singan

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, required=True)
