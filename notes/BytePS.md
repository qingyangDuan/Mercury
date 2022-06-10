# BytePS

## 0，Install BytePs

- Change gcc  to version 4.9

- Download and install cuda90.  Move cuda-9.0  to  /usr/local/cuda

- Download nccl-2.4.2 ( for cuda9.0) . then Run `make BUILDDIR=/usr/local/nccl`

- Download BytePS.

- Modify mxnet include path in `setup.py`, or use my `setup_modified_QingyangDuan.py`

- Run `sudo  python3 -m pip install .`  to install it to `/usr/local/lib/python3.7/XXX`

  or `sudo  python3 setup.py install`  to install it to `/usr/lib/python3.7/XXX`

**problems:**

- python or pip : 
  - `sudo apt install libpython3.7-dev`
  - `sudo python3 -m pip install --upgrade pip  `
  - `sudo python3 -m pip install --upgrade wheel  `
  - `sudo python3 -m pip install --upgrade setuptools  `
  
- libnccl.so.2 not found:
  
  - add path  /usr/local/nccl/lib  to LD_LIBRARY_PATH    in   .bashrc
  
- If you need to recompile BytePS:
  
  - **Delete `byteps/build`   `byteps/byteps.egg-info`  `byteps/__pycache__` .** Or it won't recompile, but just uses precious compiled  libraried in those directories.
  
  - ( Optional ) Delete installed python module in system.   e.g., in `/usr/local/lib/python3.7/dist-packages/byteps-XXX` or`/usr/lib/python3.7/dist-packages/byteps-XXX`
  - Run `sudo  python3 -m pip install ` to recompile.

## 1，BytePS for MXNet 的 push_pull 的流程  

### 1.1 对每个tensor的入队处理

- byteps/byteps/mxnet/\_\_init\_\_.py     _do_push_pull( tensor )
- byteps/byteps/mxnet/ops.py           byteps_push_pull( tensor )
- byteps/byteps/mxnet/ops.cc            byteps_mxnet_push_pull_async( tensor )
  - MXEnginePushAsync(DoPushPull  ... )   ： 把 DoPushPull  作为一个push_pull operator放入MXNet Engine Core 中。
- byteps/byteps/mxnet/ops.cc        DoPushPull(  tensor ）
  - **创建 queue_list， 包括 REDUCE，  COPYD2H（即gpu ->cpu），PUSH，PULL， COPYH2D (即 cpu-> gpu)， BROADCAST。（详情见 byteps/byteps/common/operations.cc）  **
  - EnqueueTensor(  tensor ， queue_list)
- byteps/byteps/common/operations.cc      EnqueueTensor(  tensor )
  - **PartitionTensor() ** ： 对tensor执行切分，切分成小的 task， 每个task负责一个tensor partition，每个task都保存下来它们共享的 counter_ptr。
  - BytePSGlobal::GetScheduledQueue(task->queue_list[0])->addTask(task)： 把每个 task 都放入queue_list的第一个队列中。
    - **每个 task都有queue_list ，为一个string list。  每个task的queue_list 初始化为{ REDUCE，  COPYD2H, PUSH， PULL, COPYH2D,  BROADCAST}，表示这个task的生命历程为先进入第一个全局 reduce_queue, 等待被scheduling执行完成REDUCE操作；再进入下一个全局copyd2h_queue等待本操作执行； 等等等等，直到最后一个操作完成。** 如果考虑压缩，那么queue_list 为 在 PUSH前加入COMPRESS，在 PULL后加入 DECOMPRESS。
    - 上述每个特定的全局queue都有一个异步的线程，一直根据preemption-based scheduling 的规则，从这个queue中取出最高优先级的task进行执行

### 1.2 每个queue的异步处理

 queue的统一模板，代码位置：byteps/byteps/common/scheduled_queue.cc

- **BytePSScheduledQueue::addTask() **: 把一个task入队列 q，并对队列按优先级排序
- **BytePSScheduledQueue::getTask()**： 这里会在 credits 范围内，返回最高优先级的task。
- BytePSScheduledQueue::reportFinish() ： 增加 credits 

queue的 run loop，  代码位置：byteps/byteps/common/core_loops.cc

- 用getTask 从queue中取出一个task并执行对应的操作，如reduce，push，pull
- 操作执行完成后，运行 FinishOrProceed()

函数 FinishOrProceed()， 代码位置：byteps/byteps/common/core_loops.cc

- 更新 task 信息， 
- q = queue_list[0]  （ 即找到当前执行完毕的q），执行 q->reportFinish(task->len)。  
- 删除 task->queue_list  的队首 q （ 即找到当前执行完毕的q）。
- 如果此时task->queue_list  里面还有其他操作q，就push task到这个q中去执行下一个操作。
- 如果此时task->queue_list  里面没有q了：
  - v = task->counter_ptr.get()->fetch_add(1)：即对这个 task的 counter_ptr 执行 add 1
  - if (v == (int)(task->total_partnum - 1)) 即如果 这个task 所属的父tensor的所有 子task都完成，那么执行其 complete callback （即为其父tensor的 complete callback）。往上层层传导，最后告诉MXNet Engine Core 这个 tensor的push_pull operator 已经执行完成。

### 1.3 push_queue 的异步 run loop

代码位置：byteps/byteps/common/core_loops.cc

- void PushLoop()  ： 这应该是独立的push发送线程
  - 不断执行 RunPushLoopOnce()  
- **bool RunPushLoopOnce() **： 
  - 用 getTask() 从 全局 push_queue 拿到 task， 
  - 执行 BytePSGlobal::**EncodeDefaultKey()** 为这个task选择 server 
  - 用 BytePSGlobal::GetPS()->ZPush() 调用 ps-lite 发送 task， 绑定的 complete callback是函数 FinishOrProceed()

### 1.4 pull_queue 的异步 run loop

pull 的过程与上述push类似：

代码位置：byteps/byteps/common/core_loops.cc

- void PullLoop()  ： 这应该是独立的push发送线程

  - 不断执行 RunPullLoopOnce()  

- **bool RunPullLoopOnce() **： 

  - 用 getTask() 拿到 task， 
  - 执行 BytePSGlobal::**EncodeDefaultKey()** 为这个task选择 server 
  - 用 BytePSGlobal::GetPS()->ZPull() 调用 ps-lite 发送 task， 绑定的 complete callback是函数 FinishOrProceed()

  

## 2，partition和 credit 实现在哪？

根据上述描述，调度实现在communication stack 中的位置：位于Engine Core以下，ps-lite以上（或称 ZMQ以上）。



## 3， server 的选择：实现load distribution 

代码位置：在 byteps/byteps/common/global.cc

**BytePSGlobal::EncodeDefaultKey()** ： 复现了类似于 MXNet KVStore的 EncodeDefaultKey， 即把一个task（push or pull task） 分配给一个特定的 server。 注意此时的 task 已经是tensor执行了partition之后所产生的。

- 分配算法为随机选取一个server。

- 但 当开启 BYTEPS_ENABLE_MIXED_MODE 后，即有 n个 GPU（同时也有 n 个 colocate CPU servers）和 k 个 non-colocate CPU servers，即可基于论文的section 4 的计算来实现最优分配。server的选取实现在 BytePSGlobal::Hash_Mixed_Mode 函数中。

**BytePSGlobal::Hash_Mixed_Mode() **：这个 tensor 有 (2k(n-1)) / (n^2+kn-2k) 的几率分配给 k 个 non-colocate CPU servers 中任意一个， 有 1 - (2k(n-1)) / (n^2+kn-2k)  的几率分配给 n个 colocate CPU servers 中任意一个。这个分配方法和论文 section 4 的计算结果是等价地。