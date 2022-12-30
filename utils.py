
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

# loades image given a path to image
def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

# creates a list of images with different scales
def create_pyramid(img, upfactor, min_len, max_len, match_min=True):
    scaled_imgs = []
    if type(img) == list:
        width, height = img[0].size
    else:
        width, height = img.size

    if match_min == True:
        if width <= height:
            new_width = int(min_len)
            new_height = int(min_len / width * height)
        else:
            new_height = int(min_len)
            new_width = int(min_len / height * width)
        while new_height <= max_len and new_width <= max_len:
            if type(img) == list:
                new_imgs = list()
                for i in img:
                    new_img = i.resize((new_width,new_height))
                    new_imgs.append(new_img)
                scaled_imgs.append(new_imgs)
            else:
                new_img = img.resize((new_width,new_height))
                scaled_imgs.append(new_img)
            new_width = floor(new_width * upfactor)
            new_height = floor(new_height * upfactor)
    else:
        if width <= height:
            new_height = max_len
            new_width = int(max_len / height * width)
        else:
            new_width = max_len
            new_height = int(max_len / width * height)
        while new_width >= min_len and new_height >= min_len:
            if type(img) == list:
                new_imgs = []
                for i in img:
                    new_img = i.resize((new_width, new_height))
                    new_imgs.append(new_img)
                scaled_imgs.append(new_imgs)
            else:
                new_img = img.resize((new_width,new_height))
                scaled_imgs.append(new_img)
            new_width = ceil(new_width / upfactor)
            new_height = ceil(new_height / upfactor)
        scaled_imgs = scaled_imgs[::-1]

    return scaled_imgs

# convert an image to a tensor
def convert_image2tensor(img, transform=None, device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5],
                                                             [0.5,0.5,0.5])])
    tensor = transform(img)
    tensor = tensor.unsqueeze(0)
    if device != None:
        tensor = tensor.to(device)
    return tensor

# convert a list of images to a list of tensors
def convert_images2tensors(imgs, transform=None, device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5],
                                                             [0.5,0.5,0.5])])
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
def show_tensor_image(tensor, row_n=1, use_matplot=True):
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

# def downsample(tensor,r):
#     h, w = tensor.size(-2), tensor.size(-1)
#     nh , nw = ceil(h*r), ceil(w*r)
#     tensor = F.interpolate(tensor,(nh,nw))
#     return tensor

def upsample(tensor, r):
    h, w = tensor.size(-2), tensor.size(-1)
    nh , nw = floor(h*r), floor(w*r)
    tensor = F.interpolate(tensor, size=(nh,nw))
    return tensor

class GaussianPad2d:
    def __init__(self, left, right, up, down, scale=1):
        self.left = left
        self.right = right
        self.up = up
        self.down = down
        self.scale = scale

    def __call__(self, tensor):
        x = torch.zeros(tensor.size(0), tensor.size(1),
                        tensor.size(2)+up+down,
                        tensor.size(3)+left+right,
                        device=tensor.device)
        sidepad = torch.randn(x.size(0), x.size(1),
                              left+right, x.size(3),
                              device=tensor.device) * scale
        vertpad = torch.randn(x.size(0), x.size(1),
                              x.size(2)-left-right, up+down,
                              device=tensor.device) * scale
        x[:, :, :left, :] = sidepad[:, :, :left, :]
        x[:, :, -right:, :] = sidepad[:, :, -right:, :]
        x[:, :, left:-right, :up] = vertpad[:, :, :, :up]
        x[:, :, left:-right, -down:] = vertpad[:, :, :, -down:]
        x[:, :, left:-right, up:-down] = tensor
        return x

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
