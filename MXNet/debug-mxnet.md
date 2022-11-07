# debug mxnet with ps-lite
- **编译选项：**
  Set DEBUG=1 in config.mk, so that `-g` is added to CFLAGS in Makefile。这样编译之后， libmxnet.so 文件size也会更大。    
- **Error，库文件过大：**
  When compiling, this error https://github.com/apache/mxnet/issues/17045 may happen: 
  - It seems that it is due to the code size being too large. The cause of this problem could be because we are generating code for too many architectures by default. 
  - So just manually set KNOWN_CUDA_ARCHs to build on only a few architectures. **(it's 61 for 1080Ti GPU)** And the compiling can be faster. 
- **调试模式的影响：**
 NVCC 的编译 flags 也会加入 `-G`, 这会影响cuda程序的性能，使得GPU的训练速度降低很多。  两个选择：
  - config.mk 中设置 DEBUG=0. 禁用调试模式。 
  - 手动修改Makefile, 删除 NVCC 的 `-g  -G` flags. 
