import torch
import argparse
import VGG
from model import *
from datasets import *
from torchvision.utils import save_image,make_grid
from torch.utils.data import DataLoader
import os
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def test(args):
    # parameters
    cont_img_path = args.cont_img_path
    style_img_path = args.style_img_path
    model_checkpoint = args.model_checkpoint
    vgg_checkpoint = args.vgg_checkpoint
    
    output_folder = args.output
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    device = torch.device(f"cuda:{args.cuda_dev}")
    
    # set dataset
    test_dataset = DualCamDataset(cont_img_path, style_img_path, img_size=512)
    test_loader = DataLoader(test_dataset, batch_size=1)
    print(f"test dataset length: {len(test_dataset)}")

    # initialize model and optimizer
    vgg = VGG.vgg
    vgg.load_state_dict(torch.load(vgg_checkpoint))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    net = VGG.Net(vgg).to(device)

    model = Model().to(device)
    model.load_state_dict(torch.load(model_checkpoint))

    for i, (low_cont, cont_img, style_img, low_style) in enumerate(test_loader):
        cont_img = cont_img.to(device)
        low_cont = low_cont.to(device)
        low_style = low_style.to(device)
        model.eval()
        cont_feat = net.encode_with_intermediate(low_cont)
        style_feat = net.encode_with_intermediate(low_style)

        coeffs, output = model(cont_img, cont_feat, style_feat)
        print(f"Finished {i}th image")

        save_image(output, output_folder + f'{i}_output.jpg', normalize=True)
        save_image(cont_img, output_folder + f'{i}_cont.jpg', normalize=True)
        save_image(style_img, output_folder + f'{i}_style.jpg', normalize=True)
    
    return

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Joint Bilateral learning')
    parser.add_argument('--cont_img_path', type=str, required=True, help='path to content images')
    parser.add_argument('--style_img_path', type=str, required=True, help='path to style images')
    parser.add_argument('--vgg_checkpoint', type=str, default="./checkpoints/vgg_normalised.pth",
                        help='path to style images')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                        help='path to style images')
    parser.add_argument('--output', type=str, default='./output/')
    parser.add_argument('--cuda_dev', type=int, default=0, help='cuda device')

    params = parser.parse_args()

    print('PARAMS:')
    print(params)

    test(params)