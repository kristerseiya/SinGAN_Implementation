import torch
from torch import nn
import torch.nn.functional as F
from .functions import *
from .SinGAN import *

# def loadGIF(path):
#     img = Image.open(path)
#     assert(img.is_animated)
#     n_frames = img.n_frames
#     imgs = []
#     for i in range(n_frames):
#         img.seek(i)
#         x = img.convert("RGB")
#         imgs.append(x)
#     return imgs

# class DownBlock(nn.Module):
#     def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channel,output_channel,kernel,stride,padding=padding)
#         self.conv2 = nn.Conv2d(output_channel,output_channel,kernel,stride,padding=padding)
#         self.lrelu = nn.LeakyReLU(0.2)
#         self.maxpl = nn.MaxPool2d((2,2))
#
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.lrelu(x)
#         x = self.conv2(x)
#         x = self.lrelu(x)
#         x = self.maxpl(x)
#         return x
#
# class UpBlock(nn.Module):
#     def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channel,output_channel,kernel,stride,padding=padding)
#         self.conv2 = nn.Conv2d(output_channel,output_channel,kernel,stride,padding=padding)
#         self.lrelu = nn.LeakyReLU(0.2)
#
#     def forward(self,x):
#         x = self.conv1(x)
#         x = self.lrelu(x)
#         x = self.conv2(x)
#         x = self.lrelu(x)
#         x = F.interpolate(x,scale_factor=2)
#         return x
#
# class FNet(nn.Module):
#     def __init__(self,input_channel):
#         super().__init__()
#         self.down1 = DownBlock(input_channel,32)
#         self.down2 = DownBlock(32,64)
#         self.down3 = DownBlock(64,128)
#         self.up1 = UpBlock(128,256)
#         self.up2 = UpBlock(256,128)
#         self.up3 = UpBlock(128,64)
#         self.conv1 = nn.Conv2d(64,32,(3,3),1,padding=(1,1))
#         self.conv2 = nn.Conv2d(32,2,(3,3),1,padding=(1,1))
#         self.lrelu = nn.LeakyReLU(0.2)
#         self.tanh = nn.Tanh()
#
#     def forward(self,x):
#         x = self.down1(x)
#         x = self.down2(x)
#         x = self.down3(x)
#         x = self.up1(x)
#         x = self.up2(x)
#         x = self.up3(x)
#         x = self.conv1(x)
#         x = self.lrelu(x)
#         x = self.conv2(x)
#         x = self.tanh(x) * 24.0
#         return x
#
# class ResidualBlock(nn.Module):
#     def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channel,output_channel,kernel,stride,padding=padding)
#         self.conv2 = nn.Conv2d(output_channel,output_channel,kernel,stride,padding=padding)
#         self.relu = nn.ReLU()
#
#     def forward(self,x):
#         input = x
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         return x + input
#
# class FrameGenerator(nn.Module):
#     def __init__(self,input_channel,output_channel,num_resblock):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channel,64,(3,3),1)
#         self.resblocks = nn.ModuleList()
#         for _ in range(num_resblock):
#             self.resblocks.append(ResidualBlock(64,64))
#         self.convtran1 = nn.ConvTranspose2d(64,64,(3,3),2,padding=(2,2))
#         self.convtran2 = nn.ConvTranspose2d(64,64,(3,3),2,padding=(2,2))
#         self.conv2 = nn.Conv2d(64,output_channel,(3,3),1)
#         self.relu = nn.ReLU()
#
#     def forward(self,x):
#         input = x
#         x = self.conv1(x)
#         x = self.relu(x)
#         for r in self.resblocks:
#             x = r(x)
#         x = self.convtran1(x)
#         x = x[:,:,:-1,:-1]
#         x = self.relu(x)
#         x = self.convtran2(x)
#         x = x[:,:,:-1,:-1]
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = x + F.interpolate(input[:,:,:,0:3],scale_factor=4,mode="bicubic")
#         return x
#
# class RecurrentFrameGenerator(nn.Module):
#     def __init__(self,input_channel,output_channel,num_resblock,return_seq=False):
#         super().__init__()
#         self.genF = FrameGenerator(input_channel*2,output_channel,num_resblock)
#         self.fnet = Fnet(input_channel*2)
#         self.return_seq = return_seq
#         self.gen_out_seq = []
#         self.flow_seq = []
#
#     # input size is T, B, C, H, W
#     def forward(self,x,flow):
#         NUM_FRAME = x.size(0)
#         self.gen_out_seq = []
#         self.flow_seq = []
#         if NUM_FRAME > 0:
#             input = torch.cat([x[0],torch.zeros_like(x[0])],1)
#             gen_out = self.genF(input)
#             self.gen_out_seq.append(gen_out)
#         for i in range(1,NUM_FRAME):
#             input = torch.cat([x[i-1],x[i]],1)
#             flow = self.fnet(input)
#             self.flow_seq.append(flow)
#             warped = warp(gen_out,flow)
#             input = torch.cat([x[i],warped.detach()],1)
#             gen_out = self.genF(input)
#         if return_seq:
#             return self.gen_out_seq, self.flow_seq
#         else:
#             return gen_out
#
# class FrameDiscriminator(nn.Module):
#     def __init__(self,input_channel,output_channel):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channel,3,(3,3),1,padding=(1,1))
#         self.convblocks = nn.ModuleList()
#         self.convblocks.append(ConvBatchNormLeakyBlock(3,64,kernel=(4,4),stride=2,padding=(1,1)))
#         self.convblocks.append(ConvBatchNormLeakyBlock(64,64,kernel=(4,4),stride=2,padding=(1,1)))
#         self.convblocks.append(ConvBatchNormLeakyBlock(64,128,kernel=(4,4),stride=2,padding=(1,1)))
#         self.convblocks.append(ConvBatchNormLeakyBlock(128,256,kernel=(4,4),stride=2,padding=(1,1)))
#         self.linear1 = nn.Linear()

def convertVideo2Tensor(video, transform=None, device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    video_t = []
    for v in video:
        v_t = transform(v)
        v_t = v_t.unsqueeze(0)
        if device != None:
            v_t = v_t.to(device)
        video_t.append(v_t)
    video_t = torch.stack(video_t,0)
    return video_t

def loadGIF2Tensor(path,transform=None,device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    gif = Image.open(path)
    gif_t = []
    for i in range(gif.n_frames):
        gif.seek(i)
        frame = transform(gif)
        gif_t.append(frame)
    gif_t = torch.cat(gif_t,0)
    return gif_t

def loadGIF2ndarray(path):
    gif = Image.open(path)
    frames = []
    for i in range(gif.n_frames):
        gif.seek(i)
        x = gif.convert("RGB")
        x = np.array(x)
        frames.append(x)
    frames = np.stack(frames,0)
    return frames

def calcOpticalFlowFromSeq(video):
    n_frames = len(video)
    H, W = video.shape[-3], video.shape[-2]
    grid_x, grid_y = np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
    # grid = np.stack([grid_x,grid_y],-1)
    gray = []
    for i in range(n_frames):
        g = cv2.cvtColor(video[i],cv2.COLOR_RGB2GRAY)
        gray.append(g)
    gray = np.stack(gray,0)
    flow = []
    for i in range(n_frames-1):
        f = cv2.calcOpticalFlowFarneback(gray[i], gray[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        f[:,:,0] = f[:,:,0] / W + grid_x
        f[:,:,1] = f[:,:,1] / H + grid_y
        f = np.expand_dims(f,axis=0)
        flow.append(f)
    flow = np.stack(flow,0)
    return flow

def interpolate_frames(frames,imgsize):
    new_frames = []
    for f in frames:
        f = F.interpolate(f,imgsize)
        new_frames.append(f)
    new_frames = torch.stack(new_frames,0)
    return new_frames

def interpolate_flows(flow,imgsize):
    flow = flow.permute(0,1,4,2,3)
    new_flow = []
    for f in flow:
        f = F.interpolate(f,imgsize)
        new_flow.append(f)
    new_flow = torch.stack(new_flow,0)
    new_flow = new_flow.permute(0,1,3,4,2)
    return new_flow

class FrameGenerator(nn.Module):
    def __init__(self,channel_config):
        super().__init__()
        num_conv = len(channel_config) - 1
        self.convlist = nn.ModuleList()

        if num_conv > 0:
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[0],channel_config[1],padding=num_conv))
        for i in range(1, num_conv - 1):
            self.convlist.append(ConvBatchNormLeakyBlock(channel_config[i],channel_config[i+1]))
        if num_conv > 1:
            self.convlist.append(nn.Conv2d(channel_config[-2],channel_config[-1],3,1))

    def forward(self,z,lr,prev):
        a = z + lr
        x = torch.cat([a,prev],1)
        for l in self.convlist:
            x = l(x)
        return x + lr

class RecurrentFrameGenerator(nn.Module):
    def __init__(self):
        self.gen = FrameGenerator([6,32,32,32,32,3])
        #self.fnet = FNet(6)

    def forward(self,z,lr,flow):
        NUM_FRAME = lr.size(0)
        if NUM_FRAME > 0:
            zeros = torch.zeros_like(z[0])
            gen_out = self.gen(z[0],lr[0],zeros)
            gen_out_seq = gen_out.unsqueeze(0)
        for i in range(1,NUM_FRAME):
            gen_out = F.grid_sample(gen_out,flow[i-1],align_corners=True)
            gen_out = self.gen(z[i],lr[i],gen_out)
            gen_out_seq = torch.cat([gen_out_seq,gen_out],0)
        return gen_out_seq

# class FrameCritic(nn.Module):
#     def __init__(self,channel_config):
#         super().__init__()
#         num_conv = len(channel_config) - 1
#         self.convlist = nn.ModuleList()
#
#         if num_conv > 0:
#             self.convlist.append(ConvBatchNormLeakyBlock(channel_config[0],channel_config[1],padding=num_conv))
#         for i in range(1, num_conv - 1):
#             self.convlist.append(ConvBatchNormLeakyBlock(channel_config[i],channel_config[i+1]))
#         if num_conv > 1:
#             self.convlist.append(nn.Conv2d(channel_config[-2],channel_config[-1],3,1))
#
#     def forward(self,x):
#         for l in self.convlist:
#             x = l(x)
#         return x

class TecoSinGAN():
    def __init__(self, flow, netG=None, imgsize=None, z_std=None, fixed_z=None):

        if netG == None:
            netG = []
        if imgsize == None:
            imgsize = []
        if z_std == None:
            z_std = []
        if fixed_z == None:
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
        self.flow = flow

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
        if scale==None:
            scale=self.num_scales
        with torch.no_grad():
          zeros = torch.zeros_like(self.z0[0])
          rec = self.generators[0](self.z0[0],zeros)
          for i in range(1,scale):
              rec = interpolate_frames(rec,self.imgsize[i])
              rec = self.generators[i](self.z0[i],rec)
        return rec

    def sample(self,num_sample=1,scale=None):
        if self.generators == []:
            return None
        if scale==None:
            scale=self.num_scales
        with torch.no_grad():
          zeros = torch.zeros_like(self.z0[0])
          zeros = torch.cat(num_sample*[zeros],1)
          z = self.z_std[0] * torch.randn_like(zeros)
          flow = interpolate_flows(self.flow,self.imgsize[0])
          sample = self.generators[0](z,zeros,flow)
          for i in range(1,scale):
              sample = interpolate_frames(sample,self.imgsize[i])
              flow = interpolate_flows(self.flow,self.imgsize[i])
              z = self.z_std[i] * torch.randn_like(sample)
              sample = self.generators[i](z,sample,flow)
        return sample

# reconstruction loss
# warp loss
# ping pong loss
# adversarial loss
# discriminator loss

# def WarpLoss(target,flow_seq):
#     criterion = nn.MSELoss()
#     loss = 0
#     before = target[0:-1]
#     after = target[1:]
#     for b, a, f in zip(before, after, flow_seq):
#         x = warp(b,f)
#         loss = loss + criterion(x,after)
#     return loss

def PingPongLoss(gen_out_seq):
    criterion = nn.MSELoss()
    mid = len(gen_out_seq) // 2
    loss = 0
    for i in range(mid):
        loss += criterion(gen_out_seq[i],gen_out_seq[2*mid-i])
    return loss

# def AdversarialLoss(disc_out):
#     return -disc_out.mean()
#
# def CriticLoss(disc_real,disc_fake):
#     return -disc_real.mean() + disc_fake.mean()
#
# def ReconstructionLoss(rec,target):
#     criterion = nn.MSELoss()
#     return criterion(rec,target)

def makeTriplets(seq,flow):
    n_frames = seq.size(0)
    triplets = []
    for i in range(1,n_frames-1):
        warp1 = F.grid_sample(seq[i-1],flow[i])
        warp2 = F.grid_sample(seq[i+1],flow[-i-1])
        t = torch.stack([warp1,seq[i],warp2],0)
        triplets.append(t)
    triplets = torch.stack(triplets,0)
    return triplets

def makePingPongSeq(seq):
    n_frames = seq.shape[0]
    backward = seq[0:-1]
    backward = backward[::-1]

    seq = np.concatenate([seq,backward],0)
    return seq

def showTensorFrames(seq):
    if seq.size(1) != 1:
        raise Exception("it must be 1 batch!")

    seq = seq.permute(1,0,2,3,4)
    seq = seq.squeeze(0)
    plot = torchvision.utils.make_grid(seq,8)
    plt.imshow(plot)

def TrainTecoSinGANOneScale(video,netG,netG_optim,netG_lrscheduler, \
                            netD,netD_optim,netD_lrscheduler, \
                            netG_chain,num_epoch, \
                            use_zero=True, pingpong=True, \
                            recloss_scale=10,gp_scale=0.1,z_std_scale=0.1, \
                            netG_iter=3,netD_iter=3,freq=0):

    # batch = torch.cat(batch_size*[video],1)
    imgsize = (video.size(-2),video.size(-1))
    flow = netG_chain.flow
    flow = interpolate_flows(flow,imgsize)

    if (netG_chain.num_scales == 0):
        first = True
        zeros = torch.zeros_like(video)
        # batch_zeros = torch.cat(batch_size*[zeros],1)
        z_std = z_std_scale
        fixed_z = z_std * torch.randn_like(video)
    else:
        first = False
        zeros = torch.zeros_like(video)
        # batch_zeros = torch.cat(batch_size*[zeros],1)
        prev_rec = netG_chain.reconstruct()
        prev_rec = interpolate_frames(prev_rec,imgsize)
        z_std = z_std_scale * torch.sqrt(F.mse_loss(prev_rec,video)).item()
        if use_zero:
            fixed_z = zeros
        else:
            fixed_z = z_std * torch.randn_like(video)

    netG_loss = []
    netD_loss = []

    netG.train()
    netD.train()

    for epoch in range(1,num_epoch+1):

        for i in range(netD_iter):
            netD_optim.zero_grad()

            # generate image
            if (first):
                z = z_std * torch.randn_like(zeros)
                Gout = netG(z,zeros,flow)
            else:
                z = z_std * torch.randn_like(zeros)
                base = netG_chain.sample()
                base = F.interpolate(base,imgsize)
                Gout = netG(z,base)

            triplets = makeTriplets(video,flow)
            # train critic
            Dout_real = netD(video)
            # D_loss_real = - Dout_real.mean()
            # D_loss_real.backward()
            Dout_fake = netD(Gout.detach())
            # D_loss_fake = Dout_fake.mean()
            # D_loss_fake.backward()
            D_loss = CriticLoss(Dout_real,Dout_fake)
            D_grad_penalty = GradientPenaltyLoss(netD,img,Gout.detach(),gp_scale)
            D_grad_penalty.backward()
            D_loss = D_loss.item() + D_grad_penalty.item()
            netD_optim.step()
            netD_loss.append(D_loss)

        netD_lrscheduler.step()

        disableGrad(netD)

        for i in range(netG_iter):
            if (i!=0):
                if (first):
                  Gout = netG(z,zeros,flow)
                else:
                  base = netG_chain.sample()
                  base = interpolate_frames(base,imgsize)
                  # base = F.interpolate(base,imgsize)
                  Gout = netG(z,base,flow)

            netG_optim.zero_grad()
            # train generator
            pp_loss = PingPongLoss(Gout)
            pp_loss.backward(retain_graph=True)

            Dout_fake = netD(Gout)
            adv_loss = AdversarialLoss(Dout_fake)
            # adv_loss = - Dout_fake.mean()
            adv_loss.backward()
            if (first):
              rec = netG(fixed_z,zeros,flow)
            else:
              rec = netG(fixed_z,prev_rec,flow)
            # rec_loss = F.mse_loss(rec,img) * mse_scale
            rec_loss = ReconstructionLoss(rec,video) * recloss_scale
            rec_loss.backward()
            G_loss = pp_loss.item() + adv_loss.item() + rec_loss.item()
            netG_optim.step()
            netG_loss.append(G_loss)

        netG_lrscheduler.step()
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
                  z = z_std * torch.randn_like(zeros)
                  base = netG_chain.sample()
                  # base = F.interpolate(base,imgsize)
                  base = interpolate_frames(base,imgsize)
                  sample = netG(z,base)
                  rec = netG(fixed_z,prev_rec)

            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            showTensorFrames(sample)
            plt.title("Random Sample")
            plt.subplot(1,2,2)
            showTensorFrames(rec)
            plt.title("Reconstruction")
            plt.show()
            netG.train()

    return z_std, fixed_z, (netG_loss, netD_loss)
