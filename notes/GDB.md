# GDB 调试

使用GDB调试程序，有以下两点需要注意：

1. 要使用GDB调试某个程序（如用clion调试，clion底层依赖于GDB，或手动gdb调试），该程序编译时必须加上编译选项 **`-g`**，否则该程序是不包含调试信息的；
2. GCC编译器支持 **`-O`** 和 **`-g`** 一起参与编译。GCC编译过程对进行优化的程度可分为5个等级，分别为 ：

- **-O/-O0**： 不做任何优化，这是默认的编译选项 ；
- **-O1**：使用能减少目标文件大小以及执行时间并且不会使编译时间明显增加的优化。 该模式在编译大型程序的时候会花费更多的时间和内存。在 -O1下：编译会尝试减少代 码体积和代码运行时间，但是并不执行会花费大量时间的优化操作。
- **-O2**：包含 -O1的优化并增加了不需要在目标文件大小和执行速度上进行折衷的优化。 GCC执行几乎所有支持的操作但不包括空间和速度之间权衡的优化，编译器不执行循环 展开以及函数内联。这是推荐的优化等级，除非你有特殊的需求。 -O2会比 -O1启用多 一些标记。与 -O1比较该优化 -O2将会花费更多的编译时间当然也会生成性能更好的代 码。
- **-O3**：打开所有 -O2的优化选项并且增加 -finline-functions, -funswitch-loops,-fpredictive-commoning, -fgcse-after-reload and -ftree-vectorize优化选项。这是最高最危险 的优化等级。用这个选项会延长编译代码的时间，并且在使用 gcc4.x的系统里不应全局 启用。自从 3.x版本以来 gcc的行为已经有了极大地改变。在 3.x，，-O3生成的代码也只 是比 -O2快一点点而已，而 gcc4.x中还未必更快。用 -O3来编译所有的 软件包将产生更 大体积更耗内存的二进制文件，大大增加编译失败的机会或不可预知的程序行为（包括 错误）。这样做将得不偿失，记住过犹不及。在 gcc 4.x.中使用 -O3是不推荐的。
- **-Os**：专门优化目标文件大小 ,执行所有的不增加目标文件大小的 -O2优化选项。同时 -Os还会执行更加优化程序空间的选项。这对于磁盘空间极其紧张或者 CPU缓存较小的 机器非常有用。但也可能产生些许问题，因此软件树中的大部分 ebuild都过滤掉这个等 级的优化。使用 -Os是不推荐的。

## 启动与退出

启动调试的三种方式：

1. 直接调试目标程序：gdb ./hello_server
2. **附加进程id：sudo gdb 进入，然后执行 attach pid**
3. 调试core文件：gdb filename corename

退出GDB：
- 可以用命令：**q（quit的缩写）或者 Ctr + d** 退出GDB。
- 如果GDB attach某个进程，退出GDB之前要用命令 **detach** 解除附加进程。

## 常用命令

参考：    https://zhuanlan.zhihu.com/p/297925056

- 启停
  - attach pid
  - detach
- 断点
  - b (break)  xxxx
    - **break FunctionName**，在函数的入口处添加一个断点；
  - **break LineNo**，在**当前文件**行号为**LineNo**处添加断点；
    - **break FileName:LineNo**，在**FileName**文件行号为**LineNo**处添加一个断点；
    - **break FileName:FunctionName**，在**FileName**文件的**FunctionName**函数的入口处添加断点；
    - break -/+offset，在当前程序暂停位置的前/后 offset 行处下断点；
    - break ... if cond，下条件断点；
  - info b                       # 显示断点
  
- 调试，运行，单步
  - c (continue)
  - r (run)
  - n (next)

- 查看
  - print  varName
  - ptype varName      # 查看变量类型
  - where
  - l (list)                              #  显示源码
  - set listsize count    |||     show listsize
        

- **调试模式的影响：**
 NVCC 的编译 flags 也会加入 `-G`, 这会影响cuda程序的性能，使得GPU的训练速度降低很多。  两个选择：
  - config.mk 中设置 DEBUG=0. 禁用调试模式。 
  - 手动修改Makefile, 删除 NVCC 的 `-g  -G` flags. 