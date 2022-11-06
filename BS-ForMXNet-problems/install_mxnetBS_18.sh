#! /bin/bash
# install cuda, bytescheduler and mxnet for Ubuntu 18

#FROM nvidia/cuda:9.0-cudnn7-devel




export MXNET_ROOT=/users/duanqing/incubator-mxnet
export USE_BYTESCHEDULER=1
export BYTESCHEDULER_WITH_MXNET=1
export BYTESCHEDULER_WITHOUT_PYTORCH=1
export MY_PATH="/users/duanqing"
cd $MY_PATH


# Install dev tools
sudo apt-get update && sudo apt-get install -y iputils-ping && sudo apt-get install -y apt-utils &&  sudo apt-get install -y net-tools
    # duanqingyang adding dependencies 
sudo apt-get install -y vim git python-dev build-essential &&\
    sudo apt-get install -y wget && sudo wget https://bootstrap.pypa.io/get-pip.py && sudo python get-pip.py


# install Nvidia Driver , then install CUDA Toolkit v9.0, with instructions from "https://github.com/akirademoss/cuda-9.0-installation-on-ubuntu-18.04"
sudo apt purge *nvidia* 
dpkg -l | grep nvidia  #check packages about nvidia drivers
# some packages may need “  dpkg --purge --force-all name ”  to remove them

sudo apt install nvidia-driver-450
sudo apt install nvidia-modprobe
sudo reboot
nvidia-smi 
# if it doesn't work, use cmds below according to  https://github.com/DisplayLink/evdi/issues/215
# git clone --depth 1 https://github.com/DisplayLink/evdi.git
# cd evdi
# make
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sh cuda_9.0.176_384.81_linux-run --override ##chose your own install location
edit .bashrc to add cuda/bin and cuda/lib64 to PATH and LD_LIBRARY_PATH
sudo ldconfig ..../cuda/lib64




# Install gcc 4.9
mkdir -p "$MY_PATH/gcc/" && cd "$MY_PATH/gcc/" &&\
    wget http://launchpadlibrarian.net/247707088/libmpfr4_3.1.4-1_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728424/libasan1_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728426/libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728314/gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728399/cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728404/gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728432/libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728401/g++-4.9_4.9.3-13ubuntu2_amd64.deb

cd "$MY_PATH/gcc/" &&\
    sudo dpkg -i gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
    sudo dpkg -i libmpfr4_3.1.4-1_amd64.deb &&\
    sudo dpkg -i libasan1_4.9.3-13ubuntu2_amd64.deb &&\
    sudo dpkg -i libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    sudo dpkg -i cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    sudo dpkg -i gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    sudo dpkg -i libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    sudo dpkg -i g++-4.9_4.9.3-13ubuntu2_amd64.deb

# Pin GCC to 4.9 (priority 200) to compile correctly against MXNet.
cd $MY_PATH
sudo update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
    sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
    sudo update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
    sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 && \
    sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 && \
    sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 && \
    sudo update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200

sudo pip install mxnet-cu90==1.5.0
export LD_LIBRARY_PATH="/usr/local/lib/python2.7/dist-packages/mxnet/:${LD_LIBRARY_PATH}"

# Clone MXNet as ByteScheduler compilation needs header files
git clone --recursive --branch v1.5.x https://github.com/apache/incubator-mxnet.git
cd incubator-mxnet && git reset --hard 75a9e187d00a8b7ebc71412a02ed0e3ae489d91f

# Install ByteScheduler
sudo pip install bayesian-optimization==1.0.1 six
cd /usr/local/cuda/lib64 && sudo ln -s stubs/libcuda.so libcuda.so.1
cd $MY_PATH
git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git && \
    cd byteps/bytescheduler && sudo python setup.py install 
sudo rm -f /usr/local/cuda/lib64/libcuda.so.1 && \
    sudo ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 
    # duanqingyang adding to fix mxnet running bug 

# Examples
cd "$MY_PATH/byteps/bytescheduler/examples/mxnet-image-classification"
