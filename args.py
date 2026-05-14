import argparse

def args_parser():
	parser = argparse.ArgumentParser(description='Build the splits of remote datasets')
	
	# root setting
	parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for computation (e.g., 'cuda:0', 'cuda:1', or 'cpu')")
	parser.add_argument('--data_dir', default='home/datasets/1', type=str)
	parser.add_argument('--dataset', default='AID', type=str, choices=['AID', 'NWPU-RESISC45'])
	parser.add_argument('--class_list', default='dataset/splits-0.2/classInd.txt', type=str)
	parser.add_argument('--train_list', default='dataset/splits-0.2/train_split.txt', type=str)
	parser.add_argument('--val_list', default='dataset/splits-0.2/val_split.txt', type=str)
	parser.add_argument('--ratio', default='0.2', type=str) 
	parser.add_argument('--resume_path', default='checkpoints/dataset_ratio_arch_mode.pth', type=str)

	# model setting
	parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'resnet50', 'vgg11', 'efficientnet_b3'])
	parser.add_argument('--mode', default='kd', type=str, choices=['va', 'kd'])
	parser.add_argument('--batch_size', default=24, type=int)
	parser.add_argument('--img_size', default=256, type=int)
	parser.add_argument('--n_workers', default=8, type=int)
	

	# train setting
	parser.add_argument('--start_epoch', default=0, type=int)
	parser.add_argument('--epochs', default=50, type=int)
	parser.add_argument('--step_size', default=20, type=int)
	parser.add_argument('--lr', default=1e-4, type=float)	

	# kd setting
	parser.add_argument('--temperature', default=3.0, type=float)
	args = parser.parse_args()
	return args

