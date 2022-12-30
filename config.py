
# IMAGE_TRANSFORM = transforms.Compose([transforms.ToTensor(),
#                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                       ])
import argparse

def configure():

    parser = argparse.ArgumentParser()
    parser.add_argument('--min_size', type=int, default=12)
    parser.add_argument('--max_size', type=int, default=200)
    parser.add_argument('--upfactor', type=float, default=1.33)
    parser.add_argument('--match_min', action='store_true')

    parser.add_argument('--n_epoch', type=int, default=2000)
    parser.add_argument('--recloss_coef', type=float, default=10)
    parser.add_argument('--gradpenalty_coef', type=float, default=0.1)
    parser.add_argument('--clip_range', type=float, default=0.01)
    parser.add_argument('--noise_scale', type=float, default=0.1)

    parser.add_argument('--gen_iter', type=int, default=3)
    parser.add_argument('--crt_iter', type=int, default=3)

    parser.add_argument('--mode', type=str, default='wgangp')

    parser.add_argument('--log_freq', type=int, default=0)
    parser.add_argument('--plot_freq', type=int, default=0)
    parser.add_argument('--n_sample', type=int, default=7)

    return parser
