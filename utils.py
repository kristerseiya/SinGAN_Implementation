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
import imageio

# loades image given a path to image
def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

# creates a list of images with different scales
def create_scaled_images(img,scale,min_len,max_len,match_min=True):
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

# copy image to tensor
def convert_image2tensor(img,transform=None,device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    if device != None:
        tensor = tensor.to(device)
    return tensor

# copy a list of images to a list of tensors
def convert_images2tensors(imgs,transform=None, device=None):
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

# plot tensors as images
def show_tensor_image(tensor,row_n=1,use_matplot=True):
    tensor = tensor.detach().cpu()
    tensor = torchvision.utils.make_grid(tensor,row_n)
    img = tensor.numpy()
    img = img.transpose([1,2,0])
    img = (img + 1) / 2
    img = np.clip(img,0.,1.)
    if use_matplot is True:
        plt.imshow(img)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.show()
    else:
        img = img * 255
        img = img.astype(np.uint8)
        x = Image.fromarray(img)
        x.show()



# parameter initialization
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

# turn off gradient calculation
def disable_grad(net):
    for p in net.parameters():
        p.requires_grad = False
    return

# turn on gradient calculation
def enable_grad(net):
    for p in net.parameters():
        p.requires_grad = True
    return

def downsample(tensor,r):
    h, w = tensor.size(-2), tensor.size(-1)
    nh , nw = ceil(h*r), ceil(w*r)
    tensor = F.interpolate(tensor,(nh,nw))
    return tensor

def upsample(tensor, r):
    h, w = tensor.size(-2), tensor.size(-1)
    nh , nw = floor(h*r), floor(w*r)
    tensor = F.interpolate(tensor,(nh,nw))
    return tensor

def save_tensor_image(tensor, path, row_n=2):
    tensor = tensor.detach().cpu()
    tensor = torchvision.utils.make_grid(tensor,row_n)
    arr = tensor.numpy()
    arr = arr.transpose([1,2,0])
    arr = (arr + 1) / 2
    arr = np.clip(arr,0.,1.)
    arr = arr * 255
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
    return

def save_tensor_gif(tensor, path):
    tensor = tensor.detach().cpu()
    images = tensor.numpy()
    images = images.transpose([0,2,3,1])
    images = (images + 1) / 2
    images = np.clip(images, 0., 1.)
    images = images * 255
    images = images.astype(np.uint8)
    imageio.mimsave(path,images)
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
