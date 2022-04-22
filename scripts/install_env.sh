#!/usr/bin/env bash
sudo apt-get install libglfw3
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
pip3 install -i https://mirrors.bfsu.edu.cn/pypi/web/simple cffi
pip3 install -i https://mirrors.bfsu.edu.cn/pypi/web/simple imageio pypng Cython PyOpenGL triangle glumpy pytz
pip3 install -i https://mirrors.bfsu.edu.cn/pypi/web/simple open3d
pip3 install -i https://mirrors.bfsu.edu.cn/pypi/web/simple einops
