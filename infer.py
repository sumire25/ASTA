import os
import argparse
import torch
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='asta', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--weights', type=str, required=True, help='path to model weights')
parser.add_argument('--data_dir', default='./data/', type=str)
parser.add_argument('--result_dir', default='./results/', type=str)
parser.add_argument('--dataset', default='Haze1k-thick', type=str)
args = parser.parse_args()


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


if __name__ == '__main__':
    network = eval(args.model.replace('-', '_'))()
    network.cuda()

    if os.path.exists(args.weights):
        print('==> Loading weights: ' + args.weights)
        network.load_state_dict(single(args.weights))
    else:
        print('==> Weights not found: ' + args.weights)
        exit(0)

    network.eval()

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)

    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    for batch in tqdm(test_loader, desc='Inference'):
        input_img = batch['source'].cuda()
        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input_img).clamp_(-1, 1)
            output = output * 0.5 + 0.5

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, filename), out_img)
