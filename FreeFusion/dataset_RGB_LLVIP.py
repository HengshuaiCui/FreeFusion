import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
import numpy as np
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG','bmp', 'PNG', 'gif'])


# Impervious surfaces (RGB: 255, 255, 255)
# Building (RGB: 0, 0, 255)
# Low vegetation (RGB: 0, 255, 255)
# Tree (RGB: 0, 255, 0)
# Car (RGB: 255, 255, 0)
# Clutter/background (RGB: 255, 0, 0)

# ImSurf = np.array([255, 255, 255])  # label 0
# Building = np.array([0, 0, 255]) # label 1
# LowVeg = np.array([0, 255, 255]) # label 2
# Tree = np.array([0, 255, 0]) # label 3
# Car = np.array([255, 255, 0]) # label 4
# Clutter = np.array([255, 0, 0]) # label 5
# Boundary = np.array(  [0, 0, 0]) # label 6
ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5
# Boundary = np.array([0, 0, 0]) # label 6
num_classes = 6
classes = ('ImSurf', 'Building', 'LowVeg' ,'Tree', 'Car', 'Clutter')
rgb_lable_img = [[255, 255, 255],[0, 0, 255],[0, 255, 255],[0, 255, 0],[255, 255, 0],[255, 0, 0]]


def rgb_to_2D_label(_label):  # H W C
    # label_seg = np.zeros((_label.shape[0], _label.shape[1], num_classes), dtype=np.uint8)  # 320 320
    label_seg = np.zeros((_label.shape[0], _label.shape[1]), dtype=np.uint8)#320 320
    for idex in range(num_classes):
        color = rgb_lable_img[idex]
        mask = np.all(_label == color, axis=-1)
        label_seg[mask] = idex
    return label_seg

def onehot_to_mask(img):
    mask = np.argmax(img,axis=-1)
    return mask


# def rgb_to_2D_label(_label):#H W C
#     _label = _label.transpose(2, 0, 1) # C H W 0 0 255
#
#     label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)#320 320
#     # print(label_seg.shape)
#     label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
#     label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
#
#     # label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
#
#     return label_seg

class DataLoaderTrain(Dataset):
    def __init__(self, img_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_ir_files = sorted(os.listdir(os.path.join(img_dir, 'ir', 'input')))
        tar_ir_files = sorted(os.listdir(os.path.join(img_dir, 'ir', 'target')))
        inp_rgb_files = sorted(os.listdir(os.path.join(img_dir, 'rgb', 'input')))
        tar_rgb_files = sorted(os.listdir(os.path.join(img_dir, 'rgb', 'target')))
        # tar_seg_files = sorted(os.listdir(os.path.join(img_dir, 'seg')))
        self.inp_ir_filenames = [os.path.join(img_dir, 'ir', 'input', x)  for x in inp_ir_files if is_image_file(x)]
        self.tar_ir_filenames = [os.path.join(img_dir, 'ir', 'target', x) for x in tar_ir_files if is_image_file(x)]
        self.inp_rgb_filenames = [os.path.join(img_dir, 'rgb', 'input', x) for x in inp_rgb_files if is_image_file(x)]
        self.tar_rgb_filenames = [os.path.join(img_dir, 'rgb', 'target', x) for x in tar_rgb_files if is_image_file(x)]
        # self.tar_seg_filenames = [os.path.join(img_dir, 'seg', x) for x in tar_seg_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.tar_ir_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_ir_path = self.inp_ir_filenames[index_]
        tar_ir_path = self.tar_ir_filenames[index_]
        inp_rgb_path = self.inp_rgb_filenames[index_]
        tar_rgb_path = self.tar_rgb_filenames[index_]
        # tar_seg_path = self.tar_seg_filenames[index_]

        inp_ir_img = Image.open(inp_ir_path).convert('RGB')
        tar_ir_img = Image.open(tar_ir_path).convert('RGB')
        inp_rgb_img = Image.open(inp_rgb_path).convert('RGB')
        tar_rgb_img = Image.open(tar_rgb_path).convert('RGB')
        # tar_seg_img = Image.open(tar_seg_path).convert('RGB')
        # ---------rgb_to_2D_label-----------#
        # tar_seg_array = np.array(tar_seg_img)
        # print(tar_seg_array.shape)
        # target_seg = rgb_to_2D_label(tar_seg_array)
        # --------------------#

        inp_ir_img = TF.to_tensor(inp_ir_img)
        tar_ir_img = TF.to_tensor(tar_ir_img)
        inp_rgb_img = TF.to_tensor(inp_rgb_img)
        tar_rgb_img = TF.to_tensor(tar_rgb_img)
        # tar_seg_img = torch.from_numpy(target_seg).long()
        filename = os.path.splitext(os.path.split(tar_ir_path)[-1])[0]


        return tar_ir_img, inp_ir_img, tar_rgb_img, inp_rgb_img, filename

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir):
        super(DataLoaderTest, self).__init__()
        inp_ir_files = sorted(os.listdir(os.path.join(inp_dir, 'ir')))
        inp_rgb_files = sorted(os.listdir(os.path.join(inp_dir, 'rgb')))

        self.inp_ir_filenames = [os.path.join(inp_dir, 'ir', x) for x in inp_ir_files if is_image_file(x)]
        self.inp_rgb_filenames = [os.path.join(inp_dir, 'rgb', x) for x in inp_rgb_files if is_image_file(x)]

        self.inp_ir_size = len(self.inp_ir_filenames)
        self.inp_rgb_size = len(self.inp_rgb_filenames)



    def __len__(self):
        return self.inp_ir_size

    def __getitem__(self, index):

        path_ir_inp = self.inp_ir_filenames[index]
        path_rgb_inp = self.inp_rgb_filenames[index]

        filename = os.path.splitext(os.path.split(path_ir_inp)[-1])[0]
        inp_ir = Image.open(path_ir_inp).convert('RGB')
        inp_rgb = Image.open(path_rgb_inp).convert('RGB')

        inp_ir = TF.to_tensor(inp_ir)
        inp_rgb = TF.to_tensor(inp_rgb)

        return inp_ir, inp_rgb, filename
