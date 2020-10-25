from PIL import Image
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def loadImage(path):
    img = Image.open(path)
    return img

def createScaledImages(img,scale,min_len,max_len):
    scaled_imgs = []
    width, height = img.size
    if width <= height:
        new_width = (int)(min_len)
        new_height = (int)(min_len / width * height)
    else:
        new_height = (int)(min_len)
        new_width = (int)(min_len / height * width)
    while (new_height <= max_len and new_width <= max_len):
        new_img = img.resize((new_width,new_height))
        scaled_imgs.append(new_img)
        new_width, new_height = (int)(new_width / scale), (int)(new_height / scale)

    return scaled_imgs

def convertImage2Tensor(img,device,transform=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    tensor = transform(img)
    tensor = tensor.to(device)
    return tensor

def convertImages2Tensors(imgs,device,transform=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transformed_imgs = []
    for img in imgs:
        tensor = transform(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)
        transformed_imgs.append(tensor)
    return transformed_imgs

def showTensorImage(tensor):
    img = tensor.detach().cpu().numpy()
    if img.dim() == 4 && img.size(0) == 1:
      img = img.squeeze(0)
    img = img.transpose([1,2,0])
    img = (img + 1) / 2
    img[img>1] = 1
    img[img<0] = 0
    plt.imshow(img)

def xavier_uniform_weight_init(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight)
    return

def xavier_normal_weight_init(layer):
    if type(layer) == nn.Conv2d:
        torch.nn.init.xavier_normal_(layer.weight)
    return

def disableGrad(net):
    for p in net.parameters():
        p.requires_grad = False
    return

def enableGrad(net):
    for p in net.parameters():
        p.requires_grad = True
    return

def warp(x,flow):
    B,C,H,W = x.size()
    grid_y, grid_x = torch.meshgrid(torch.range(-1,1,2/(H-1)),torch.range(-1,1,2/(W-1)))
    identity = torch.stack([grid_x, grid_y],0)
    identity = identity.unsqueeze(0)
    relative_flow = flow + identity
    warped = F.grid_sample(x, relative_flow.permute(0,2,3,1),align_corners=True)
    return warped
