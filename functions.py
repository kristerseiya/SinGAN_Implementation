from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torch.nn.functional as F

def loadPILImage(path):
    img = Image.open(path)
    return img

def createScaledImgs(img,scale,min_len):
    scaled_imgs = []
    scaled_imgs.append(img)
    width, height = img.size
    width, height = (int)(width * scale), (int)(height * scale)
    while (height > min_len and width > min_len):
        new_img = img.resize((width,height))
        scaled_imgs.append(new_img)
        width, height = (int)(width * scale), (int)(height * scale)

    return scaled_imgs[::-1]

def loadToTensor(imgs,transform_img,device):
    transformed_imgs = []
    tsfm = transforms.Compose([transforms.ToTensor()])
    for img in imgs:
        tensor = transform_img(tsfm(img))
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)
        transformed_imgs.append(tensor)
    return transformed_imgs

def showTensorImg(tensor):
  img = tensor.detach().cpu().numpy()
  img = img.squeeze(0)
  img = img.transpose([1,2,0])
  img = (img + 1) / 2
  img[img>1] = 1
  img[img<0] = 0
  plt.imshow(img)

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
