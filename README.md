# local Flow Matching


Codes to implement Local FM as in the paper "Local Flow Matching Generative Models", arXiv version: [arXiv:2410.02548](https://arxiv.org/abs/2410.02548).

The codes for 2D and tabular datasets are adapted from the implementation at [this repository](https://github.com/hamrel-cxu/LocalFlowMatching) by Chen Xu. The current repository also includes image experiments. 

Contributors: [xycheng](https://github.com/xycheng), [hamrel-cxu](https://github.com/hamrel-cxu/), [echen333](https://github.com/echen333)


## Environment

```bash
conda env create -f environment.yml
```

To active 

```bash
conda activate lfm
```


## 2D toy data

To run the code,

```bash
# Tree
python main_2d.py --hyper_param_config config/tree_l.yaml
```
The distribution uses `./data/img_tree.png`.

## Tabular data

Data download and set up: 
  
Follow 
https://github.com/gpapamak/maf?tab=readme-ov-file#how-to-get-the-datasets 
regarding data downloading. Specifically, the following should suffice

```bash
wget https://zenodo.org/record/1161203/files/data.tar.gz
tar -xzf data.tar.gz
rm -r data/mnist
rm -r data/cifar10
rm -r data/hepmass
```

The unzipped `data` folder should be structured as `{TASK}/file`:
- BSDS300/BSDS300.hdf5
- gas/ethylene_CO.pickle
- miniboone/data.npy
- power/data.npy

Run `python check_tabular_stats.py` to ensure data can be loaded, and it will print the statistics of each dataset. 

To run the code,

```bash
# BSDS300
python main_tabular.py --hyper_param_config config/bsds300_l.yaml 
```

You can also specify a seed by adding `--seed {seed}` at the end.

To run the code on the other three datasets,

```bash
# GAS
python main_tabular.py --hyper_param_config config/gas_l.yaml
# MINIBOONE
python main_tabular.py --hyper_param_config config/miniboone_l.yaml
# POWER
python main_tabular.py --hyper_param_config config/power_l.yaml
```

# Images

The Unet architecture is from the OpenAI guided diffusion [repository](https://github.com/openai/guided-diffusion). 

## CIFAR-10

To run the code,

```bash
python main_cifar.py --hyper_param_config config/cifar_l.yaml
```

Data will be downloaded when first running the code, stored in `./data`.

We use clean-fid, the cached statistics will be saved as an .npz file in the local folder of the site packages, e.g., `~/.conda/envs/lfm/lib/python3.10/site-packages/cleanfid/stats/`. Same for Flowers and Imagenet32.

## Flowers 

To run the code,

```bash
python main_flowers.py --hyper_param_config config/flowers_l.yaml 
```
Data will be downloaded when first running the code, stored in `./data`.

## Imagenet32

Download and unzip the imagenet32 data inside `./data`

```bash
cd ./data/
wget https://image-net.org/data/downsample/Imagenet32_train.zip
wget https://image-net.org/data/downsample/Imagenet32_val.zip
unzip -q Imagenet32_train.zip
unzip -q Imagenet32_val.zip
```

To run the code, go back to the root folder, and run

```bash
python main_imagenet32.py --hyper_param_config config/imagenet_l.yaml 
```
