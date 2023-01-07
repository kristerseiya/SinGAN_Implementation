import torch
# from SinGAN_Implementation import models, utils, train
from SINGAN import models, utils, train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help="input image")
parser.add_argument('--r', default=0.75, help="scale of downsampling")
parser.add_argument('--min_len', default=12, help="minimum height or width of image")
parser.add_argument('--max_len', default=250, help="maximum height or width of image")
parser.add_argument('--match_min', default=True, help="match the minimum length")

arg = parser.parse_args()

img = load_image(args.images)
scaled_img_list = utils.create_scaled_images(img,arg.r,arg.min_len,arg.max_len,match_min=arg.match_min)

print("{} scales created".format(len(scaled_img_list)))

if torch.cuda.is_available() is True:
    print("Using CUDA")
    device = torch.device('cuda:0')
else:
    device = None

print("training starts...")
singan = train.train(arg.r, scaled_img_list, device=device)

cpu = torch.device('cpu')
singan = singan.to(cpu)
models.save_singan(singan,'singan.pth')
