from PIL import Image

def loadImg(path):
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

def preprocess(imgs,transform_img,device):
    transformed_imgs = []
    for img in imgs:
        tensor = transform_img(img)
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
