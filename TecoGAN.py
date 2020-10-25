import torch
from torch import nn

class DownBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel,stride,padding=padding)
        self.conv2 = nn.Conv2d(output_channel,output_channel,kernel,stride,padding=padding)
        self.lrelu = nn.LeakyReLU(0.2)
        self.maxpl = nn.MaxPool2d((2,2))

    def forward(self,x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.maxpl(x)
        return x

class UpBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel,stride,padding=padding)
        self.conv2 = nn.Conv2d(output_channel,output_channel,kernel,stride,padding=padding)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x = F.interpolate(x,scale_factor=2)
        return x

class FNet(nn.Module):
    def __init__(self,input_channel):
        super().__init__()
        self.down1 = DownBlock(input_channel,32)
        self.down2 = DownBlock(32,64)
        self.down3 = DownBlock(64,128)
        self.up1 = UpBlock(128,256)
        self.up2 = UpBlock(256,128)
        self.up3 = UpBlock(128,64)
        self.conv1 = nn.Conv2d(64,32,(3,3),1,padding=(1,1))
        self.conv2 = nn.Conv2d(32,2,(3,3),1,padding=(1,1))
        self.lrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.tanh(x) * 24.0
        return x

class ResidualBlock(nn.Module):
    def __init__(self,input_channel,output_channel,kernel=3,stride=1,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,output_channel,kernel,stride,padding=padding)
        self.conv2 = nn.Conv2d(output_channel,output_channel,kernel,stride,padding=padding)
        self.relu = nn.ReLU()

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + input

class FrameGenerator(nn.Module):
    def __init__(self,input_channel,output_channel,num_resblock):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,64,(3,3),1)
        self.resblocks = nn.ModuleList()
        for _ in range(num_resblock):
            self.resblocks.append(ResidualBlock(64,64))
        self.convtran1 = nn.ConvTranspose2d(64,64,(3,3),2,padding=(2,2))
        self.convtran2 = nn.ConvTranspose2d(64,64,(3,3),2,padding=(2,2))
        self.conv2 = nn.Conv2d(64,output_channel,(3,3),1)
        self.relu = nn.ReLU()

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.relu(x)
        for r in self.resblocks:
            x = r(x)
        x = self.convtran1(x)
        x = x[:,:,:-1,:-1]
        x = self.relu(x)
        x = self.convtran2(x)
        x = x[:,:,:-1,:-1]
        x = self.relu(x)
        x = self.conv2(x)
        x = x + F.interpolate(input[:,:,:,0:3],scale_factor=4,mode="bicubic")
        return x

class RecurrentFrameGenerator(nn.Module):
    def __init__(self,input_channel,output_channel,num_resblock,return_seq=False):
        super().__init__()
        self.genF = FrameGenerator(input_channel*2,output_channel,num_resblock)
        self.fnet = Fnet(input_channel*2)
        self.return_seq = return_seq
        self.gen_out_seq = []
        self.flow_seq = []

    # input size is T, B, C, H, W
    def forward(self,x,flow):
        NUM_FRAME = x.size(0)
        self.gen_out_seq = []
        self.flow_seq = []
        if NUM_FRAME > 0:
            input = torch.cat([x[0],torch.zeros_like(x[0])],1)
            gen_out = self.genF(input)
            self.gen_out_seq.append(gen_out)
        for i in range(1,NUM_FRAME):
            input = torch.cat([x[i-1],x[i]],1)
            flow = self.fnet(input)
            self.flow_seq.append(flow)
            warped = warp(gen_out,flow)
            input = torch.cat([x[i],warped.detach()],1)
            gen_out = self.genF(input)
        if return_seq:
            return self.gen_out_seq, self.flow_seq
        else:
            return gen_out

class FrameDiscriminator(nn.Module):
    def __init__(self,input_channel,output_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel,3,(3,3),1,padding=(1,1))
        self.convblocks = nn.ModuleList()
        self.convblocks.append(ConvBatchNormLeakyBlock(3,64,kernel=(4,4),stride=2,padding=(1,1)))
        self.convblocks.append(ConvBatchNormLeakyBlock(64,64,kernel=(4,4),stride=2,padding=(1,1)))
        self.convblocks.append(ConvBatchNormLeakyBlock(64,128,kernel=(4,4),stride=2,padding=(1,1)))
        self.convblocks.append(ConvBatchNormLeakyBlock(128,256,kernel=(4,4),stride=2,padding=(1,1)))
        self.linear1 = nn.Linear()

# reconstruction loss
# warp loss
# ping pong loss
# adversarial loss
# discriminator loss

def WarpLoss(target,flow_seq)

def PingPongLoss(gen_out_seq):
    criterion = nn.MSELoss()
    mid = len(gen_out_seq) // 2
    forward = gen_out_seq[1:mid-1]
    backward = gen_out_seq[-2:mid+1:-1]
    loss = 0
    for out1, out2 in zip(forward,backward):
        loss += criterion(out1,out2)
    return loss

class RecurrentGenerator(nn.Module):
    def __init__(self):
        self.gen = SinGenerator([6,32,32,32,32,3])
        self.fnet = FNet(3)

    def forward(self,x):
        NUM_FRAME = x.size(0)
        gen_out_seq = torch.tensor(0)
        flow_seq = torch.tensor(0)
        if NUM_FRAME > 0:
            input = torch.cat([x[0],torch.zeros_like(x[0])],1)
            gen_out = self.gen(input)
            gen_out_seq = torch.cat([gen_out_seq,gen_out],0)
        for i in range(1,NUM_FRAME):
            input = torch.cat([x[i-1],x[i]],1)
            flow = self.fnet(input)
            flow_seq = torch.cat([flow_seq,flow],0)
            warped = warp(gen_out,flow)
            input = torch.cat([x[i],warped],1)
            gen_out = self.gen(input)
        return gen_out_seq, flow_seq
