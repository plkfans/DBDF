import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from args import args_parser
from datasets import load_datasets
from utils import *
from PIL import Image
import cv2
import numpy as np
from time import time
import pdb
from model import *
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

args = args_parser()
best_acc = [0, 0, 0, 0, 0]

def label2cls(root, dataset, class_path):
    f = open(os.path.join(root, class_path.replace('dataset', dataset)), 'r')
    line = f.readline()
    label2cls_list ={}   
    while line:
        line = line.strip('\n')
        cls, label = line.split(' ')
        label2cls_list[str(label)] = str(cls)
        line = f.readline()
    return label2cls_list


def plot_confusion_matrix(dataset, true_list, pred_list, label2cls_list, mode):
    save_path = './save_status'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    labels = []
    for key, value in label2cls_list.items():
        labels.append(value)
    tick_marks = np.float32(np.array(range(len(labels)))) + 0.5

    cm = confusion_matrix(true_list, pred_list)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    # # UCM and RSSCN7
    # fontsize_axis = 5.2
    # fontsize_prop = 4.2

    # AID and NWPU-RESISC45
    fontsize_axis = 4.2
    fontsize_prop = 3.53

    barsize = 5
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_norm[y_val][x_val]
        if c > 0.01:
            color="white" if c > 0.5 else "black"
            plt.text(x_val, y_val, '%0.2f'%(c,), color=color, fontsize=fontsize_prop, va='center', ha='center')

    plt.gca().set_xticks(tick_marks)
    plt.gca().set_yticks(tick_marks)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-', linewidth=0.3)

    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues) 
    xlocations = np.array(range(len(labels))) 
    plt.xticks(xlocations, labels, fontsize=fontsize_axis, rotation=270) 
    plt.yticks(xlocations, labels, fontsize=fontsize_axis) 

    ax = plt.gca()
    n = len(labels)
    ax.hlines(y=np.arange(-0.5, n + 0.5, 1), xmin=-0.5, xmax=n - 0.5,
              colors='gray', linewidth=0.5, alpha=0.7)
    ax.vlines(x=np.arange(-0.5, n + 0.5, 1), ymin=-0.5, ymax=n - 0.5,
              colors='gray', linewidth=0.5, alpha=0.7)

    cb = plt.colorbar(shrink=1.0) 
    cb.ax.tick_params(labelsize=barsize)

    plt.tight_layout()
    plt.savefig('./save_status/confusion_matrix_' + dataset + '_' + mode + '.pdf', format='pdf')

if __name__ == '__main__':  

    # load device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # load datasets feat
    train_list = args.train_list.replace('dataset', args.dataset)
    val_list = args.val_list.replace('dataset', args.dataset)
   
    
    train_loader, val_loader = load_datasets(args.data_dir, 
                                             train_list, 
                                             val_list, 
                                             args.mode, 
                                             args.batch_size, 
                                             args.img_size, 
                                             args.n_workers)

    
    if args.dataset=='AID':
        n_classes = 30
    elif args.dataset=='NWPU-RESISC45':
        n_classes = 45

    resume_path = args.resume_path.replace('dataset', args.dataset)  \
                                .replace('ratio', args.ratio)   \
                                .replace('arch', args.arch)   \
                                .replace('mode', args.mode) 
    # bulid model
    net = Model(arch=args.arch, n_classes=n_classes, mode=args.mode).to(device)

    if os.path.exists(resume_path):
        if args.mode == "va":
            print ('Load model')
            resume = torch.load(resume_path, map_location='cpu')
            net.load_state_dict(resume['state_dict'], strict=False)
            # optim_1.load_state_dict(resume['optim_1'])
            
            # sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size, last_epoch = resume['epoch'])
            # args.start_epoch = resume['epoch'] + 1
            best_acc[0] = resume['acc']
            # print('current epoch: ', resume['epoch'])
            # print('Restored epoch: ', args.start_epoch)
            # print('current learning rate: ', optim_1.param_groups[0]['lr'])
            print('current best_acc: {:.2f}%' .format(best_acc[0]*100))
        if args.mode == "kd":
            print ('Load model')
            resume = torch.load(resume_path, map_location='cpu')
            net.load_state_dict(resume['state_dict'], strict=False)
            # optim_1.load_state_dict(resume['optim_1'])
            # args.start_epoch = resume['epoch'] + 1
            # sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size, last_epoch=resume['epoch'])
            best_acc = resume['acc']

            # print('optim_1 current learning rate: ', optim_1.param_groups[0]['lr'])
            # print('current epoch: ', resume['epoch'])
            # print('Restored epoch: ', args.start_epoch)
            print ('Curr_Global_Acc: {:.2f}%  Curr_Local_Acc: {:.2f}%  Curr_En_Acc: {:.2f}%  Curr_Avg_Acc: {:.2f}%' 
				.format(best_acc[1]*100, best_acc[2]*100, best_acc[3]*100, best_acc[4]*100))


    net = net.eval()

    label2cls_list = label2cls(args.data_dir, args.dataset, args.class_list)

    if args.mode == "va":
        pred_list = []
        label_list = []

        beg_time = time()
        with torch.no_grad():
            for filenames, imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                preds = net(imgs)
                _, preds = torch.max(preds, dim=1)

                pred_list.extend(preds.cpu().numpy())
                label_list.extend(labels.cpu().numpy())

        
        plot_confusion_matrix(args.dataset, label_list, pred_list, label2cls_list, args.mode)
        end_time = time()
        print("Total time:", end_time - beg_time)

    if args.mode == "kd":
        pred_list = []
        label_list = []

        beg_time = time()
        
        for filenames, imgs, labels in val_loader:
            with torch.no_grad():
                imgs = imgs.to(device)
                labels = labels.to(device)
                [logits_g, logits_l, logits_en], _ = net(imgs)

                preds = logits_en
                _, preds = torch.max(preds, dim=1)
                pred_list.extend(preds.cpu().numpy())
                label_list.extend(labels.cpu().numpy())

        plot_confusion_matrix(args.dataset, label_list, pred_list, label2cls_list, args.mode)
        end_time = time()
        print("Total time:", end_time - beg_time)

