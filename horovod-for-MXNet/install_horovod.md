-  首先，修复 mxnet-1.5/include 里的软连接，使得其指向 3rdparty 中制定的位置。  
   - mxnet-1.5 用 git clone 下载之后，其 /include中的软连接损坏。这些软连接用于horovod编译时查找头文件。
- pip3 install --user horovod==0.18.0   // for mxnet-1.5.x
