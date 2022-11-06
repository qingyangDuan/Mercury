# debug mxnet with ps-lite
- Set DEBUG=1 in config.mk, so that -g is added to CFLAGS when compiling   
- When compiling, this error https://github.com/apache/mxnet/issues/17045 may happen: 
  - It seems that it is due to the code size being too large. The cause of this problem could be because we are generating code for too many architectures by default. 
  - So just manually set KNOWN_CUDA_ARCHs to build on only a few architectures. (it's 61 for 1080Ti GPU) And the compiling can be faster.

