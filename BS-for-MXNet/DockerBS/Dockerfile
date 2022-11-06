# create an image for bytescheduler with mxnet

FROM nvidia/cuda:9.0-cudnn7-devel

ENV MY_PATH="/users/duanqing"
ENV MXNET_ROOT=/users/duanqing/incubator-mxnet

ENV USE_BYTESCHEDULER=1
ENV BYTESCHEDULER_WITH_MXNET=1
ENV BYTESCHEDULER_WITHOUT_PYTORCH=1

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"




WORKDIR $MY_PATH

# Install dev tools
RUN apt-get update && apt-get install -y iputils-ping && apt-get install -y apt-utils && apt-get install -y net-tools
    # duanqingyang adding dependencies 
    
RUN apt-get install -y vim git python-dev build-essential &&\
    apt-get install -y wget && wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

# Install gcc 4.9
RUN mkdir -p "$MY_PATH/gcc/" && cd "$MY_PATH/gcc/" &&\
    wget http://launchpadlibrarian.net/247707088/libmpfr4_3.1.4-1_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728424/libasan1_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728426/libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728314/gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728399/cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728404/gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728432/libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    wget http://launchpadlibrarian.net/253728401/g++-4.9_4.9.3-13ubuntu2_amd64.deb

RUN cd "$MY_PATH/gcc/" &&\
    dpkg -i gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libmpfr4_3.1.4-1_amd64.deb &&\
    dpkg -i libasan1_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i cpp-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i gcc-4.9_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb &&\
    dpkg -i g++-4.9_4.9.3-13ubuntu2_amd64.deb

# Pin GCC to 4.9 (priority 200) to compile correctly against MXNet.
RUN update-alternatives --install /usr/bin/gcc gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc $(readlink -f $(which gcc)) 100 && \
    update-alternatives --install /usr/bin/g++ g++ $(readlink -f $(which g++)) 100 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ $(readlink -f $(which g++)) 100

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-gcc x86_64-linux-gnu-gcc /usr/bin/gcc-4.9 200 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.9 200 && \
    update-alternatives --install /usr/bin/x86_64-linux-gnu-g++ x86_64-linux-gnu-g++ /usr/bin/g++-4.9 200

RUN pip install mxnet-cu90==1.5.0

# Clone MXNet as ByteScheduler compilation needs header files
RUN git clone --recursive --branch v1.5.x https://github.com/apache/incubator-mxnet.git
RUN cd incubator-mxnet && git reset --hard 75a9e187d00a8b7ebc71412a02ed0e3ae489d91f

# Install ByteScheduler
RUN pip install bayesian-optimization==1.0.1 six
RUN cd /usr/local/cuda/lib64 && ln -s stubs/libcuda.so libcuda.so.1
RUN git clone --branch bytescheduler --recursive https://github.com/bytedance/byteps.git && \
    cd byteps/bytescheduler && python setup.py install
RUN rm -f /usr/local/cuda/lib64/libcuda.so.1 && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1 
    # duanqingyang adding to fix mxnet running bug 

# Examples
WORKDIR "$MY_PATH/byteps/bytescheduler/examples/mxnet-image-classification"
