import torch
from .functions import *
from .SinGAN import *

def main():

    fixed_z = 0.1 * torch.randn_like(scaled_img_tensor_list[0])
    imgsize_list = [(x.size[1],x.size[0]) for x in scaled_img_list]
    netG_chain = SinGeneratorChain(fixed_z)
    netD_list = []

    num_scales = len(scaled_img_list)
    num_layers = 5
    min_num_channel = 32
    max_num_channel = 32
    num_channel = 32
    input_channel = 3
    z_scale = 0.1

    for i in range(num_scales):

        netG = Generator([input_channel,num_channel,num_channel,num_channel,3])
        netG = netG.to(cuda)
        netG.apply(xavier_normal_init)
        netD = Critic([3,num_channel,num_channel,num_channel,num_channel,1])
        netD = netD.to(cuda)
        netD.apply(xavier_normal_init)
        netG_optim = torch.optim.Adam(netG.parameters(),lr=5e-4,betas=(0.5,0.999))
        netD_optim = torch.optim.Adam(netD.parameters(),lr=5e-4,betas=(0.5,0.999))
        netG_lrscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=netG_optim,milestones=[1600],gamma=0.1)
        netD_lrscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=netD_optim,milestones=[1600],gamma=0.1)
        img = scaled_img_tensor_list[i]

        print("Scale {}".format(i+1))

        z_std, _ = TrainSinGANOneScale(img,netG,netG_optim,netG_lrscheduler,
                                       netD,netD_optim,netD_lrscheduler,
                                       netG_chain,2000,
                                       freq=200,batch_size=1,
                                       mse_scale=10,gp_scale=0.1,z_std_scale=z_scale,
                                       netG_iter=3,netD_iter=3)

        netD.eval()
        disableGrad(netD)
        netD_list.append(netD)
        netG.eval()
        disableGrad(netG)
        imgsize = (img.size(-2),img.size(-1))
        netG_chain.append(netG,imgsize,z_std)

        if (i+1) % 4 == 0:
            num_channel = num_channel * 2


if __name__ == __main__:
    main()
