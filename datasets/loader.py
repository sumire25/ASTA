import os
import re
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def _extract_id(filename):
	match = re.match(r'(\d+)', filename)
	return match.group(1) if match else filename


def _build_id_map(directory):
	return {_extract_id(f): f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))}


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


def _detect_subfolder_names(root_dir):
	for src_name, tgt_name in [('hazy', 'GT'), ('input', 'target')]:
		if os.path.isdir(os.path.join(root_dir, src_name)) and os.path.isdir(os.path.join(root_dir, tgt_name)):
			return src_name, tgt_name
	raise FileNotFoundError(f"Could not find image pair folders in {root_dir}. Expected 'hazy'/'GT' or 'input'/'target'.")


class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		src_name, tgt_name = _detect_subfolder_names(self.root_dir)
		self.src_name = src_name
		self.tgt_name = tgt_name

		src_map = _build_id_map(os.path.join(self.root_dir, self.src_name))
		tgt_map = _build_id_map(os.path.join(self.root_dir, self.tgt_name))

		self.common_ids = sorted(set(src_map.keys()) & set(tgt_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
		self.src_map = src_map
		self.tgt_map = tgt_map
		self.img_num = len(self.common_ids)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		img_id = self.common_ids[idx]
		source_img = read_img(os.path.join(self.root_dir, self.src_name, self.src_map[img_id])) * 2 - 1
		target_img = read_img(os.path.join(self.root_dir, self.tgt_name, self.tgt_map[img_id])) * 2 - 1

		if self.mode == 'train':
			[source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

		if self.mode == 'valid':
			[source_img, target_img] = align([source_img, target_img], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': self.tgt_map[img_id]}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = sorted(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		img_name = self.img_names[idx]
		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

		return {'img': hwc_to_chw(img), 'filename': img_name}
