<div align="center">

# pvnet_lightning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is PVNet rewritten using pytorch lightning.
And project template is used [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template.git)
Here is the official code for [PVNet](https://github.com/zju3dv/clean-pvnet)

## How to run

### Install dependencies

```bash
# clone project
git clone https://github.com/weiwei11/pvnet_lightning.git
cd pvnet_lightning

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

# or use bash
bash script/install_env.sh
```

### Compile cuda extensions under `lib/csrc`:
```
ROOT=/path/to/clean-pvnet

cd $ROOT/lib/csrc

export CUDA_HOME="/usr/local/cuda-10.2"
cd ../ransac_voting
python setup.py build_ext --inplace
cd ../nn
python setup.py build_ext --inplace
cd ../fps
python setup.py build_ext --inplace

# If you want to use the uncertainty-driven PnP
cd ../uncertainty_pnp
sudo apt-get install libgoogle-glog-dev
sudo apt-get install libsuitesparse-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libceres-dev
sudo apt-get install libgoogle-glog-dev 
python setup.py build_ext --inplace
```
or 
```bash
bash build_csrc.sh
```

### Set up datasets:
```
ROOT=/path/to/pvnet_lightning
cd $ROOT/data
ln -s /path/to/linemod linemod
ln -s /path/to/linemod_orig linemod_orig
ln -s /path/to/occlusion_linemod occlusion_linemod
```

### Train

```bash
# train model
python train.py experiment=pvnet obj_cls='cat'
```

### Test

```bash
# test model
python test.py experiment=pvnet obj_cls='cat'
```

### Download datasets which are formatted for this project:
1. [linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EXK2K0B-QrNPi8MYLDFHdB8BQm9cWTxRGV9dQgauczkVYQ?e=beftUz)
2. [linemod_orig](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EaoGIPguY3FAgrFKKhi32fcB_nrMcNRm8jVCZQd7G_-Wbg?e=ig4aHk): The dataset includes the depth for each image.
3. [occlusion linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/ESXrP0zskd5IvvuvG3TXD-4BMgbDrHZ_bevurBrAcKE5Dg?e=r0EgoA)
4. [truncation linemod](https://1drv.ms/u/s!AtZjYZ01QjphfuDICdni1IIM4SE): Check [TRUNCATION_LINEMOD.md](TRUNCATION_LINEMOD.md) for the information about the Truncation LINEMOD dataset.
5. [Tless](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/EsKEY3aHNElEjaKbhCJVyQgBUGTlprdcyF5sgLjEv8J8TQ?e=NbJpkM): `cat tlessa* | tar xvf - -C .`.
6. [Tless cache data](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EWf-M5HRcH1JnBNN9yE1a84BYNAU7x1DoU_-W3Onl5Xxog?e=HZSrMu): It is used for training and testing on Tless.
7. [SUN2012pascalformat](http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz)

## Note

1. PyCharm users will need to enable “emulate terminal” in output console option in run/debug configuration to see styled output. Reference: https://rich.readthedocs.io/en/latest/introduction.html#requirements
