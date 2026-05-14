import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
from pycm import *
from args import args_parser
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import os


args = args_parser()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def train(epoch, train_loader, net, optim_1, criterion, mode):
    print('train at epoch {}'.format(epoch))

    net.train()
    
    losses = AverageMeter()

    g_accuracies = AverageMeter()
    l_accuracies = AverageMeter()
    en_accuracies = AverageMeter()
    avg_accuracies = AverageMeter()
    accuracies = AverageMeter()

    if mode == 'va':
        for i, (filenames, imgs_s1, labels) in enumerate(train_loader):
            imgs_s1 = Variable(imgs_s1.to(device))    
            labels = Variable(labels.to(device))
            logits = net(imgs_s1)

            optim_1.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optim_1.step()
            
            acc = accuracy(logits, labels)
            losses.update(loss.item(), logits.size(0))        
            accuracies.update(acc, logits.size(0))

            if (i%50==0 and i!=0) or i+1==len(train_loader):
                current_lr = optim_1.param_groups[0]['lr']
                print ('Train:  Epoch[{}]:{}/{}  Learning Rate: {}  Loss:{:.4f}   Accu:{:.2f}%'.\
                        format(epoch, i, len(train_loader), current_lr, float(losses.avg), float(accuracies.avg)*100))
                
    elif mode == 'kd':
        for i, (filenames, imgs_s1, labels) in enumerate(train_loader):
            imgs_s1 = Variable(imgs_s1.to(device))    
            labels = Variable(labels.to(device))


            logits, [local_imgs, masks, masks_new, bboxes] = net(imgs_s1)
            logits_avg = sum(logits[:])/len(logits)
            Teacher_logits = logits_avg.detach()

            optim_1.zero_grad()
            loss = torch.FloatTensor([0.]).to(device)
            for index in range(len(logits)):
                loss += criterion(logits[index], labels)
                loss += KD_loss(logits[index], Teacher_logits) 
            loss.backward()
            optim_1.step()
            
            g_acc = accuracy(logits[0], labels)
            l_acc = accuracy(logits[1], labels)
            en_acc = accuracy(logits[2], labels)
            avg_acc = accuracy(Teacher_logits, labels)

            losses.update(loss.item(), Teacher_logits[0].size(0))   

            g_accuracies.update(g_acc, Teacher_logits[0].size(0))
            l_accuracies.update(l_acc, Teacher_logits[0].size(0))
            en_accuracies.update(en_acc, Teacher_logits[0].size(0))
            avg_accuracies.update(avg_acc, Teacher_logits[0].size(0))

            if (i%50==0 and i!=0) or i+1==len(train_loader):
                # Get the learning rate from the optimizer
                current_lr_1 = optim_1.param_groups[0]['lr']
                
                print ('Train:   Epoch[{}]:{}/{}   Loss:{:.4f}   LR:{}  Global_Acc:{:.2f}%  Local_Acc:{:.2f}%'
                        '   En_Acc:{:.2f}%   Avg_Acc:{:.2f}%' .\
                        format(epoch, i, len(train_loader), float(losses.avg),  current_lr_1, 
                            float(g_accuracies.avg)*100,
                            float(l_accuracies.avg)*100, 
                            float(en_accuracies.avg)*100, 
                            float(avg_accuracies.avg)*100))
                
    return accuracies.avg


def validation(epoch, best_acc, val_loader, net, optim_1, resume_path, criterion, mode):
	print('val at epoch {}'.format(epoch))
	net.eval()

	losses = AverageMeter()
	g_accuracies = AverageMeter()
	l_accuracies = AverageMeter()
	en_accuracies = AverageMeter()
	avg_accuracies = AverageMeter()


	accuracies = AverageMeter()

	if mode == 'va':
		for i, (filenames, imgs_s1, labels) in enumerate(val_loader):
			with torch.no_grad():
				img_s1 = Variable(imgs_s1.to(device))     
				labels = Variable(labels.to(device))
				logits = net(img_s1)
				loss = criterion(logits, labels)
				acc = accuracy(logits, labels)
			losses.update(loss.item(), logits.size(0))
			accuracies.update(acc, logits.size(0))

			if (i%50==0 and i!=0) or i+1==len(val_loader):
				print ('Validation:   Epoch[{}]:{}/{}    Loss:{:.4f}   Accu:{:.2f}%'.   \
						format(epoch, i, len(val_loader), float(losses.avg), float(accuracies.avg)*100))
		
		if accuracies.avg >= best_acc[0]:
			best_acc[0] = accuracies.avg
			save_file_path = resume_path
			states = {'state_dict': net.state_dict(),
					'epoch':epoch,
					'optim_1': optim_1.state_dict(),
					'acc':best_acc[0]}
			torch.save(states, save_file_path)
			print ('Saved!')
		print ('curr_acc: {:.2f}%'.format(accuracies.avg*100))
		print ('best_acc: {:.2f}%'.format(best_acc[0]*100))
		
		
	
	elif mode == 'kd':
		for i, (filenames, imgs_s1, labels) in enumerate(val_loader):
			with torch.no_grad():
				imgs_s1 = Variable(imgs_s1.to(device))  
				labels = Variable(labels.to(device))

				logits, [local_imgs, masks, masks_new, bboxes] = net(imgs_s1)
				logits_avg = sum(logits[:])/len(logits)
				Teacher_logits = logits_avg.detach()

				loss = torch.FloatTensor([0.]).to(device)
				for index in range(len(logits)):
					loss += criterion(logits[index], labels)
					loss += KD_loss(logits[index], Teacher_logits) 
					
			g_acc = accuracy(logits[0], labels)
			l_acc = accuracy(logits[1], labels)
			en_acc = accuracy(logits[2], labels)
			avg_acc = accuracy(Teacher_logits, labels)

			losses.update(loss.item(), Teacher_logits[0].size(0))

			g_accuracies.update(g_acc, Teacher_logits[0].size(0))
			l_accuracies.update(l_acc, Teacher_logits[0].size(0))
			en_accuracies.update(en_acc, Teacher_logits[0].size(0))
			avg_accuracies.update(avg_acc, Teacher_logits[0].size(0))
			

			if (i%50==0 and i!=0) or i+1==len(val_loader):
				print ('Validation:   Epoch[{}]:{}/{}    Loss:{:.4f}   Global_Acc:{:.2f}%  Local_Acc:{:.2f}%'
					'   En_Acc:{:.2f}%   Avg_Acc:{:.2f}%'.\
						format(epoch, i, len(val_loader), float(losses.avg),
								float(g_accuracies.avg)*100,
								float(l_accuracies.avg)*100, 
								float(en_accuracies.avg)*100, 
								float(avg_accuracies.avg)*100))
					
		print ('Curr_Global_Acc: {:.2f}%  Curr_Local_Acc: {:.2f}%  Curr_En_Acc: {:.2f}%  Curr_Avg_Acc: {:.2f}% '
		.format((g_accuracies.avg)*100, (l_accuracies.avg)*100, (en_accuracies.avg)*100, (avg_accuracies.avg)*100))
		
		if g_accuracies.avg >= best_acc[1]:
			best_acc[1] = g_accuracies.avg
		
		if l_accuracies.avg >= best_acc[2]:
			best_acc[2] = l_accuracies.avg

		if en_accuracies.avg >= best_acc[3]:
			best_acc[3] = en_accuracies.avg

		if avg_accuracies.avg >= best_acc[4]:
			best_acc[4] = avg_accuracies.avg
			save_file_path = resume_path
			states = {'state_dict': net.state_dict(),
					'epoch':epoch,
					'optim_1': optim_1.state_dict(),
					'acc':best_acc}
			torch.save(states, save_file_path)
			print ('Saved!')

		print ('Best_Global_Acc: {:.2f}%  Best_Local_Acc: {:.2f}%  Best_En_Acc: {:.2f}%  Best_Avg_Acc: {:.2f}%' 
				.format(best_acc[1]*100, best_acc[2]*100, best_acc[3]*100, best_acc[4]*100))
		

	return best_acc, accuracies.avg

 