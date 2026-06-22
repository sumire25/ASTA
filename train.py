import os
import csv
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from pytorch_msssim import ssim

from utils import AverageMeter
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='asta', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=True, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--dataset', default='Haze1k-thick', type=str, help='dataset name')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
parser.add_argument('--val_freq', default=3, type=int, help='validation frequency in epochs')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def calculate_sam(img1, img2):
    eps = 1e-8
    dot = (img1 * img2).sum(dim=1)
    norm1 = img1.pow(2).sum(dim=1).sqrt()
    norm2 = img2.pow(2).sum(dim=1).sqrt()
    cos = dot / (norm1 * norm2 + eps)
    cos = cos.clamp(-1, 1)
    sam = torch.acos(cos).mean()
    return torch.degrees(sam).item()


def calculate_ergas(img1, img2, ratio=1):
    mean_ref = img2.mean(dim=(2, 3), keepdim=True)
    rmse = (img1 - img2).pow(2).mean(dim=(2, 3), keepdim=True).sqrt()
    ergas = 100 / ratio * ((rmse / (mean_ref + 1e-8)).pow(2).mean(dim=1)).sqrt().mean()
    return ergas.item()


def train(train_loader, network, criterion, optimizer, scaler):
	losses = AverageMeter()

	torch.cuda.empty_cache()
	
	network.train()

	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with autocast(args.no_autocast):
			output = network(source_img)
			loss = criterion(output, target_img)

		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()
	SSIM_meter = AverageMeter()
	SAM_meter = AverageMeter()
	ERGAS_meter = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		with torch.no_grad():
			output = network(source_img).clamp_(-1, 1)

		output_norm = output * 0.5 + 0.5
		target_norm = target_img * 0.5 + 0.5

		mse_loss = F.mse_loss(output_norm, target_norm, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))

		_, _, H, W = output_norm.size()
		down_ratio = max(1, round(min(H, W) / 256))
		ssim_val = ssim(F.adaptive_avg_pool2d(output_norm, (int(H / down_ratio), int(W / down_ratio))),
						F.adaptive_avg_pool2d(target_norm, (int(H / down_ratio), int(W / down_ratio))),
						data_range=1, size_average=True).item()
		SSIM_meter.update(ssim_val, source_img.size(0))

		sam_val = calculate_sam(output_norm, target_norm)
		SAM_meter.update(sam_val, source_img.size(0))

		ergas_val = calculate_ergas(output_norm, target_norm)
		ERGAS_meter.update(ergas_val, source_img.size(0))

	return PSNR.avg, SSIM_meter.avg, SAM_meter.avg, ERGAS_meter.avg


if __name__ == '__main__':
	setting_filename = os.path.join('config.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)

	network = eval(args.model)()
	network = network.cuda()

	criterion = nn.L1Loss()

	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer") 

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'], eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])
	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'], 
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

	save_dir = args.save_dir
	os.makedirs(save_dir, exist_ok=True)

	csv_path = os.path.join(save_dir, 'training_metrics.csv')
	with open(csv_path, 'w', newline='') as f:
		writer = csv.writer(f)
		writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_sam', 'val_ergas'])

	if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
		print('==> Start training, current model name: ' + args.model)

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.model))

		best_psnr = 0
		for epoch in tqdm(range(setting['epochs'] + 1)):
			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()

			if epoch % args.val_freq == 0:
				avg_psnr, avg_ssim, avg_sam, avg_ergas = valid(val_loader, network)

				print(f'\n[Val Epoch {epoch}] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SAM: {avg_sam:.4f} | ERGAS: {avg_ergas:.4f}')

				writer.add_scalar('valid_psnr', avg_psnr, epoch)
				writer.add_scalar('valid_ssim', avg_ssim, epoch)
				writer.add_scalar('valid_sam', avg_sam, epoch)
				writer.add_scalar('valid_ergas', avg_ergas, epoch)

				with open(csv_path, 'a', newline='') as f:
					csv_writer = csv.writer(f)
					csv_writer.writerow([epoch, loss, avg_psnr, avg_ssim, avg_sam, avg_ergas])

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					torch.save({'state_dict': network.state_dict()},
                			   os.path.join(save_dir, args.model+'.pth'))
				
				writer.add_scalar('best_psnr', best_psnr, epoch)

	else:
		print('==> Existing trained model')
		exit(1)
