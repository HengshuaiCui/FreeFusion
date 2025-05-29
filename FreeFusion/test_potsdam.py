import kornia
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB_potsdam import get_test_data
from MMFNet import MMFNet
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def rgb_to_ycbcr(img):
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr
def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [204, 102, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [85, 167, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [0, 255, 255]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [153, 102, 153]
    return mask_rgb

parser = argparse.ArgumentParser(description='Image Fusion using FreeFusion')
parser.add_argument('--input_dir', default='./dataset/test/', type=str, help='Directory for results')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/potsdam/model_potsdam.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--num_classes', default=6, type=int)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model = MMFNet(args.num_classes)

utils.load_checkpoint(model,args.weights)
print("===>Testing using weights: ",args.weights)
model.cuda()
model = nn.DataParallel(model)
model.eval()

datasets = ['potsdam']

for dataset in datasets:
    img_dir_test = os.path.join(args.input_dir, dataset)
    test_dataset = get_test_data(img_dir_test)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

    result_dir  = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            inp_ir = data_test[0].cuda()
            inp_rgb = data_test[1].cuda()
            filenames = data_test[2]

            input_hsv = kornia.color.rgb_to_hsv(inp_rgb)
            _, _, _, _, _, fus = model(inp_ir[:, :1, :, :], input_hsv[:, 2:, :, :])
            fus_hsv = torch.cat((input_hsv[:, 0:1, :, :], input_hsv[:, 1:2, :, :], torch.clamp(fus, 0, 1)),
                                    dim=1)
            fus_rgb = kornia.color.hsv_to_rgb(fus_hsv)
            # -------Using this code in quantitative comparisons------------#
            # input_ycbcr = rgb_to_ycbcr(inp_rgb)
            # _, _, _, _, _, fus = model(inp_ir[:, :1, :, :], input_ycbcr[:, :1, :, :])
            # fus_ycbcr = torch.cat(
            #     (torch.clamp(fus, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
            #     dim=1)
            # fus_rgb = kornia.color.ycbcr_to_rgb(fus_ycbcr)
            # --------------------------------------------------------------#
            ones = torch.ones_like(fus)
            zeros = torch.zeros_like(fus)
            fus_rgb = torch.where(fus_rgb > ones, ones, fus_rgb)
            fus_rgb = torch.where(fus_rgb < zeros, zeros, fus_rgb)
            fus_rgb = fus_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()  # B C H W->B H W C
            fus_rgb = (fus_rgb - np.min(fus_rgb)) / (
                    np.max(fus_rgb) - np.min(fus_rgb)
            )
            fus_rgb = np.uint8(255.0 * fus_rgb)

            for batch in range(len(fus)):
                fus_img = fus_rgb[batch]
                utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), fus_img)


