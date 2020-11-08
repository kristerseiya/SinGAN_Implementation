from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil, floor

def loadImage(path):
    img = Image.open(path)
    return img

def loadGIF(path):
    img = Image.open(path)
    assert(img.is_animated)
    n_frames = img.n_frames
    imgs = []
    for i in range(n_frames):
        img.seek(i)
        x = img.convert("RGB")
        imgs.append(x)
    return imgs

def createScaledImages(img,scale,min_len,max_len,match_min=True):
    scaled_imgs = []
    if type(img) == list:
        width, height = img[0].size
    else:
        width, height = img.size

    if match_min == True:
        if width <= height:
            new_width = (int)(min_len)
            new_height = (int)(min_len / width * height)
        else:
            new_height = (int)(min_len)
            new_width = (int)(min_len / height * width)
        while new_height <= max_len and new_width <= max_len:
            if type(img) == list:
                new_imgs = []
                for i in img:
                    new_img = i.resize((new_width,new_height))
                    new_imgs.append(new_img)
                scaled_imgs.append(new_imgs)
            else:
                new_img = img.resize((new_width,new_height))
                scaled_imgs.append(new_img)
            new_width, new_height = floor(new_width / scale), floor(new_height / scale)
    else:
        if width <= height:
            new_height = max_len
            new_width = (int)(max_len / height * width)
        else:
            new_width = max_len
            new_height = (int)(max_len / width * height)
        while new_width >= min_len and new_height >= min_len:
            if type(img) == list:
                new_imgs = []
                for i in img:
                    new_img = i.resize((new_width,new_height))
                    new_imgs.append(new_img)
                scaled_imgs.append(new_imgs)
            else:
                new_img = img.resize((new_width,new_height))
                scaled_imgs.append(new_img)
            new_width, new_height = ceil(new_width * scale), ceil(new_height * scale)
        scaled_imgs = scaled_imgs[::-1]

    return scaled_imgs

def convertImage2Tensor(img,transform=None,device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    if device != None:
        tensor = tensor.to(device)
    return tensor

def convertImages2Tensor(imgs,transform=None, device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transformed_imgs = []
    for img in imgs:
        if type(img) == list:
            tensor = []
            for i in img:
                t = transform(i)
                tensor.append(t)
            tensor = torch.stack(tensor,0)
        else:
            tensor = transform(img)
            tensor = tensor.unsqueeze(0)
        if device != None:
            tensor = tensor.to(device)
        transformed_imgs.append(tensor)
    return transformed_imgs

def showTensorImage(tensor,row_n=1):
    # if tensor.dim() == 4 and tensor.size(0) == 1:
    #     tensor = tensor.squeeze(0)
    # elif tensor.dim() == 4:
    #     idx = random.choice(range(tensor.size(0)))
    #     tensor = tensor[idx]
    tensor = tensor.detach().cpu()
    tensor = torchvision.utils.make_grid(tensor,row_n)
    img = tensor.numpy()
    img = img.transpose([1,2,0])
    img = (img + 1) / 2
    img[img>1] = 1
    img[img<0] = 0
    plt.imshow(img)

def xavier_uniform_weight_init(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.1)
    return

def xavier_normal_weight_init(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.1)
    return

def disableGrad(net):
    for p in net.parameters():
        p.requires_grad = False
    return

def enableGrad(net):
    for p in net.parameters():
        p.requires_grad = True
    return


## SSIM
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
