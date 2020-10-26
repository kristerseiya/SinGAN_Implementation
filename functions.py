from PIL import Image
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2

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

def convertImage2Tensor(img,transform=None,device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    tensor = transform(img)
    if device != None:
        tensor = tensor.to(device)
    return tensor

def convertImages2Tensors(imgs,transform=None, device=None):
    if transform==None:
        transform = transforms.Compose([transforms.ToTensor(), \
                                        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transformed_imgs = []
    for img in imgs:
        tensor = transform(img)
        tensor = tensor.unsqueeze(0)
        if device != None:
            tensor = tensor.to(device)
        transformed_imgs.append(tensor)
    return transformed_imgs

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

def showTensorImage(tensor):
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy()
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

# def warp(x,flow):
#     B,C,H,W = x.size()
#     grid_y, grid_x = torch.meshgrid(torch.range(-1,1,2/(H-1)),torch.range(-1,1,2/(W-1)))
#     identity = torch.stack([grid_x, grid_y],0)
#     identity = identity.unsqueeze(0)
#     relative_flow = flow + identity
#     warped = F.grid_sample(x, relative_flow.permute(0,2,3,1),align_corners=True)
#     return warped

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
