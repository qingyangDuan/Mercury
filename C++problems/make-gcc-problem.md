# 1）如果我们要用gdb去debug（如用clion的gdb调试）, 要用 -g 的 cflags     
   - 对于MXNet来说，要在 config.mk 中设置 DEBUG=1， 即在make时会加入 -g   

# 2）原 .cc 文件需要 include 的一些自定义的 .h 等 header 文件，那么 .h 的位置需要添加到 -Ipath 中    
# 3）原 .cc 文件需要 load 的一些自定义的 .so 或 .a 等库文件，那么库文件的位置需要添加到 -Lpath 中    

例如：  
CFLAGS = -g   
LDFLAGS = -L/home/duanqingyang/libvma/src/vma/.libs   
INFLAGS = -I/home/duanqingyang/libvma/src/vma/.libs   

g++ $(INFLAGS) $(LDFLAGS) $(CFLAGS) -o client client.cc  #-lvma     



# 4) 查看可执行文件或 .o 文件的动态依赖库   
ldd file   

# 5） 目标文件(ELF）的三种类型：   
https://zhuanlan.zhihu.com/p/71372182   
