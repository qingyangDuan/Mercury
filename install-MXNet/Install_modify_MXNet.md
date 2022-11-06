# 版本：
mxnet: 1.5.0 (cuda: 9.0)  
BS:   
gluonnlp: 0.8.3

# Install MXNet
## a) install mxnet from pip
 python3 -m pip install --user mxnet-cu90==1.5.0  // MXNet lib will appear at ~/.local/lib/python2.7/site-packages/mxnet

## b) install mxnet from source: 
- **1) pre-requisite**:     
      sudo apt-get install build-essential  
      sudo apt install libopencv-dev -y  
      sudo apt-get install libatlas-base-dev  -y  
      sudo apt install libopenblas-dev -y  
      install nvidia-driver-460 , install cuda-9.0  
      sudo vim /etc/ld.so.conf #add path: cuda-9.0/lib64  
      sudo ldconfig
 - **2) 下载**： git clone --recursive --branch v1.5.x https://github.com/apache/incubator-mxnet.git
 - **3) 编译**：
     - cd incubator-mxnet
     - 把incubator-mxnet-1.5/make/config.mk文件复制到incubator-mxnet-1.5/ 内指导编译，
     - 修改config.mk内部的几个选项数值： USE_DIST_KVSTORE = 1，USE_MKLDNN = 0, USE_CUDA=1, USE_CUDA_PATH = XXX（CUDA文件夹具体位置）。
     - 修改 Makefile 中 KNOWN_CUDA_ARCHS 为只支持当前 GPU 的 cuda_arch，可大幅提高编译速度。如1080Ti GPU arch 为 61。
     - 执行 make 命令
 - **3.1) 调试 Debug**:
     - 如果要用gdb调试，要在 config.mk 中设置 DEBUG=1， 即在make时会加入 `-g`   
 - **4) 安装**：`python3 -m pip install --user -e ./python`  
Note that the `-e` flag is optional. It is equivalent to --editable,   
and means that if you edit the source files, these changes will be reflected in the package installed.  
`--user` means installing to your own account python lib space.

# Install gluonnlp from source:
- cd gluonnlp
- python3 -m pip install --user -e  . (or python3 setup.py install --user)

# Modify MXNet python lib:
Modify Mxnet python code in ~/incubator-mxnet/python.   
If u install MXNet with -e, then the mxnet python lib in XXX/lib/python(2 or 3)  is link to  /home/duanqingyang/incubator-mxnet/python.   
So u  don't need to do python install again.

# Modify MXNet c++ lib:
Modify Mxnet c++ code in ~/incubator-mxnet/src .  
Make again so to update libmxnet.so in ~/incubator-mxnet/lib, which is used by mxnet python lib in ~/incubator-mxnet/python
