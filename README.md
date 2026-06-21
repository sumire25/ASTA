# Additional Self-attention Transformer with Adapter for Thick Haze Removal

> **Abstract:** 
Remote sensing images (RSIs) are widely used in the fields of geological resources monitoring, earthquake relief, and weather forecasting, but they are easily nullified due to haze cover. Transformer-based image dehazing model can better remove the haze in RSIs and improve the clarity of RSIs. However, due to the insufficient ability to extract detailed information, the model performs poorly in the case of thick haze. To solve this problem, this letter introduces an additional self-attention (AS) mechanism to help the model acquire more detailed information based on the existing Transformer-based image dehazing model and introduces an adapter module to improve the model’s fitting capacity with newly added content. Experimental results on benchmark RSIs indicate that the proposed method yields an average improvement of 0.95 in peak signal-to-noise ratio (PSNR) and 0.6% in structural similarity index metrices (SSIM) for light haze removal. Notably, the method exhibits a significant enhancement of 1.34 in PSNR and 1.9% in SSIM for the removal of thick haze, underscoring its advantage in heavy haze conditions.

## Preparation

### Install

We conduct all the experiments on Python 3.7 + PyTorch 1.10.2 + CUDA 11.3

1. Conda environment and dependencies
```
conda create -n asta python=3.7
conda activate asta
conda install pytorch=1.10.2 torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Download

All the pretrained models can be obtained on [BaiduPan](https://pan.baidu.com/s/1C0j4vaVthGgMklGXce4JSA)(y9s4)

The final file path should be the same as the following:

```
┬─ save_models
│   ├─ asta.pth
│   └─ ... (model name)
└─ data
    ├─ Haze1k-thin
    │   ├─ train
    │   │   ├─ GT
    │   │   │   └─ ... (image filename)
    │   │   └─ hazy
    │   │       └─ ... (corresponds to the former)
    │   └─ test
    │       └─ ...
    └─ ... (dataset name)
```

## Training and Evaluation

### Train

Training settings can be modified in `config.json`
Then run the following script to train the model:

```sh
python train.py --dataset (dataset name)
```

For example, we train the ASTA on the Haze1k-thin:

```sh
python train.py --dataset Haze1k-thin
```

### Test

Run the following script to test the trained model:

```sh
python test.py --dataset (dataset name)
```

For example, we test the ASTA on the Haze1k-thin:

```sh
python test.py --dataset Haze1k-thin
```

If you have downloaded all the saved models, the test scripts can be found in `run.sh`.

## Citation
```
@ARTICLE{10443626,
  author={Cai, Zhenyang and Ning, Jin and Ding, Zhiheng and Duo, Bin},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Additional Self-Attention Transformer With Adapter for Thick Haze Removal}, 
  year={2024},
  volume={21},
  number={},
  pages={1-5},
  keywords={Adaptation models;Computational modeling;Transformers;Feature extraction;Atmospheric modeling;Task analysis;Data mining;Image dehazing;remote sensing image (RSI);thick haze;transformer},
  doi={10.1109/LGRS.2024.3368430}}
```
