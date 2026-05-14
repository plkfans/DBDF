import os
import torch
import torch.nn as nn
from args import args_parser
from train_val import train, validation
from datasets import load_datasets
from collections import OrderedDict
from model import *
from time import time
import numpy as np


args = args_parser()
best_acc = [0, 0, 0, 0, 0]

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

    # bulid model
    resume_path = args.resume_path.replace('dataset', args.dataset)  \
                                .replace('ratio', args.ratio)   \
                                .replace('arch', args.arch)   \
                                .replace('mode', args.mode)   

    if args.dataset=='AID':
        n_classes = 30
    elif args.dataset=='NWPU-RESISC45':
        n_classes = 45

    net = Model(arch=args.arch, n_classes=n_classes, mode=args.mode).to(device)
                    

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')    

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optim_1 = torch.optim.Adam(net.get_parameters(), lr=args.lr)
    
    if os.path.exists(resume_path):
        if args.mode == "va":
            print ('Load model')
            resume = torch.load(resume_path)
            net.load_state_dict(resume['state_dict'], strict=False)
            optim_1.load_state_dict(resume['optim_1'])
            
            sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size, last_epoch = resume['epoch'])
            args.start_epoch = resume['epoch'] + 1
            best_acc[0] = resume['acc']
            print('current epoch: ', resume['epoch'])
            print('Restored epoch: ', args.start_epoch)
            print('current learning rate: ', optim_1.param_groups[0]['lr'])
            print('current best_acc: {:.2f}%' .format(best_acc[0]*100))
        if args.mode == "kd":
            print ('Load model')
            resume = torch.load(resume_path)
            net.load_state_dict(resume['state_dict'], strict=False)
            optim_1.load_state_dict(resume['optim_1'])
            args.start_epoch = resume['epoch'] + 1
            sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size, last_epoch=resume['epoch'])
            best_acc = resume['acc']

            print('optim_1 current learning rate: ', optim_1.param_groups[0]['lr'])
            print('current epoch: ', resume['epoch'])
            print('Restored epoch: ', args.start_epoch)
            print ('Curr_Global_Acc: {:.2f}%  Curr_Local_Acc: {:.2f}%  Curr_En_Acc: {:.2f}%  Curr_Avg_Acc: {:.2f}%' 
				.format(best_acc[1]*100, best_acc[2]*100, best_acc[3]*100, best_acc[4]*100))
            
    else:
        sche_1 = torch.optim.lr_scheduler.StepLR(optim_1, step_size=args.step_size)

    # train model
    train_time = 0
    val_time = 0
    all_time = 0
    for i in range(args.start_epoch, args.epochs):

        beg_time = time()
        train_acc = train(i, train_loader, net, optim_1, criterion, args.mode)
        end_time = time()
        train_time = train_time + (end_time - beg_time)
        all_time = all_time + (end_time - beg_time)
        print ('training_time: ', train_time)

        beg_time = time()
        best_acc, val_acc = validation(i, best_acc, val_loader, net, optim_1, resume_path, criterion, args.mode)
        end_time = time()
        val_time = val_time + (end_time - beg_time)
        all_time = all_time + (end_time - beg_time)
        print ('validation_time: ', val_time)
        print ('all_time: ', all_time)
        print ('')	

        sche_1.step()
