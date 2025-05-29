import os
import kornia
from config import Config
opt = Config('training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import utils
from data_RGB_mfnet import get_training_data
from MMFNet import MMFNet
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import torch.utils.data
from utils.seg_util import *
from utils.dice import *
import warnings
from utils.MEF_SSIM_loss import th_SSIM_LOSS,Y_Upper
import logging
def rgb_to_ycbcr(img):
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr


# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 0
    data = opt.Datasets.data
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, data, 'models')
    utils.mkdir(model_dir)
    train_dir = opt.TRAINING.TRAIN_DIR

    ######### Model ###########
    model = MMFNet(opt.TRAINING.NUM_CLASSES)
    model.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    new_lr = opt.OPTIM.LR_INITIAL
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, 'model_latest.pth')
        utils.load_checkpoint(model, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest)
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    ######### Loss ###########
    l1_criterion = nn.L1Loss()

    CE_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    Dice_criterion = DiceLoss(smooth=0.05, ignore_index=255)

    ######### DataLoaders ###########
    train_dataset = get_training_data(train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                              drop_last=True, pin_memory=True)#drop_last=False

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            # zero_grad
            for param in model.parameters():
                param.grad = None

            target_ir = data[0].cuda()
            input_ir = data[1].cuda()
            target_rgb = data[2].cuda()
            input_rgb = data[3].cuda()
            target_seg = data[4].cuda()

            input_ycbcr = rgb_to_ycbcr(input_rgb)
            # ------------------
            target = Y_Upper(target_ir[:, :1, :, :], input_ycbcr[:, :1, :, :], 1.7).cuda()
            # ------------------
            #----model output result----#
            res_ir, res_y, res_rec_ir, res_rec_y, res_seg, _ = model(input_ir[:, :1, :, :], input_ycbcr[:, :1, :, :])

            res_ycbcr = torch.cat((torch.clamp(res_y,0,1),input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
                                  dim=1)
            res_rgb = kornia.color.ycbcr_to_rgb(res_ycbcr)

            res_rec_ycbcr = torch.cat((torch.clamp(res_rec_y, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
                                  dim=1)
            res_rec_rgb = kornia.color.ycbcr_to_rgb(res_rec_ycbcr)

            loss_ir = l1_criterion(res_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_ir)
            loss_rgb = l1_criterion(res_rgb, target_rgb) + th_SSIM_LOSS(target, res_y)
            loss_rec_ir = l1_criterion(res_rec_ir, target_ir[:, :1, :, :]) + th_SSIM_LOSS(target, res_rec_ir)
            loss_rec_rgb = l1_criterion(res_rec_rgb, target_rgb) + th_SSIM_LOSS(target, res_rec_y)
            loss_seg = Dice_criterion(res_seg, target_seg) + CE_criterion(res_seg, target_seg)#to(torch.int64)

            loss = loss_ir + loss_rgb + loss_rec_ir + loss_rec_rgb + loss_seg

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))
