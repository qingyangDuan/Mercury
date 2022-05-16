# BytePS

## 1，BytePS for MXNet 的 push_pull 的流程  

- byteps/byteps/mxnet/\_\_init\_\_.py     _do_push_pull( tensor )
- byteps/byteps/mxnet/ops.py           byteps_push_pull( tensor )
- byteps/byteps/mxnet/ops.cc            byteps_mxnet_push_pull_async( tensor )
  - MXEnginePushAsync(DoPushPull  ... )   ： 把 DoPushPull  作为一个push_pull operator放入MXNet Engine Core 中。
  - DoPushPull(  tensor ）
- byteps/byteps/common/operations.cc      EnqueueTensor(  tensor )
  - PartitionTensor()  ： 对tensor执行切分，切分成小的 task， 每个task负责一个tensor partition，每个task都保存下来它们共享的 counter_ptr。
    - **每个 task都有queue_list ，为一个string list。  每个task的queue_list 初始化为{ PUSH， PULL}，表示这个task的生命历程为 先进入全局 push_q, 完成后再进入 全局 pull_q。如果考虑压缩，那么queue_list 为 { COMPRESS，PUSH， PULL， DECOMPRESS}。**
  - BytePSGlobal::GetScheduledQueue(task->queue_list[0])->addTask(task)： 把每个 task 都放入queue_list的第一个队列中，即全局 push_q。





## 2，partition和 credit 实现在哪？

调度实现在communication stack 中的位置：位于Engine Core以下，ps-lite以上（或称 ZMQ以上）。

### 2.1 Queue的统一模板

代码位置：byteps/byteps/common/scheduled_queue.cc

- **BytePSScheduledQueue::addTask() **: 把一个task入队列 q，并对队列按优先级排序

- **BytePSScheduledQueue::getTask()**： 这里会在 credits 范围内，返回最高优先级的task。
- BytePSScheduledQueue::reportFinish() ： 增加 credits 



### 2.2 Push过程

代码位置：byteps/byteps/common/core_loops.cc

- void PushLoop()  ： 这应该是独立的push发送线程
  - 不断执行 RunPushLoopOnce()  
- **bool RunPushLoopOnce() **： 
  - 用 getTask() 拿到 task， 
  - 执行 BytePSGlobal::**EncodeDefaultKey()** 为这个task选择 server 
  - 用 BytePSGlobal::GetPS()->ZPush() 调用 ps-lite 发送 task， 绑定的 complete callback是函数 FinishOrProceed()
- void FinishOrProceed()：
  - 更新 task 信息， 
  - 执行 push_q->reportFinish(task->len)
  - 把 task->queue_list  的队首 queue入口删除，此时里面还有queue入口，就把task在加入这个queue中。实际上就是task 又进入了 全局 pull_q 中

### 2.3 Pull 过程

pull 的过程与上述push类似：

代码位置：byteps/byteps/common/core_loops.cc

- void PullLoop()  ： 这应该是独立的push发送线程
  - 不断执行 RunPullLoopOnce()  
- **bool RunPullLoopOnce() **： 
  - 用 getTask() 拿到 task， 
  - 执行 BytePSGlobal::**EncodeDefaultKey()** 为这个task选择 server 
  - 用 BytePSGlobal::GetPS()->ZPull() 调用 ps-lite 发送 task， 绑定的 complete callback是函数 FinishOrProceed()
- void FinishOrProceed()：
  - 更新 task 信息， 
  - q = queue_list[0]  : 找到当前的q，即全局 pull_q
  - 执行 q->reportFinish(task->len)  
  - 把 task->queue_list  的队首 q删除，此时里面没有q即为空了。
    - v = task->counter_ptr.get()->fetch_add(1)：即对这个 task的 counter_ptr 执行 add 1
    - if (v == (int)(task->total_partnum - 1)) 即如果 这个task 所属的父tensor的所有 子task都完成，那么执行其 complete callback （即为其父tensor的 complete callback）。往上层层传导，最后告诉MXNet Engine Core 这个 tensor的push_pull operator 已经执行完成。



## 3， server 的选择：实现load distribution 

代码位置：在 byteps/byteps/common/global.cc

**BytePSGlobal::EncodeDefaultKey()** ： 复现了类似于 MXNet KVStore的 EncodeDefaultKey， 即把一个task（push or pull task） 分配给一个特定的 server。 注意此时的 task 已经是tensor执行了partition之后所产生的。

- 分配算法为随机选取一个server。

- 但 当开启 BYTEPS_ENABLE_MIXED_MODE 后，即有 n个 GPU（同时也有 n 个 colocate CPU servers）和 k 个 non-colocate CPU servers，即可基于论文的section 4 的计算来实现最优分配。server的选取实现在 BytePSGlobal::Hash_Mixed_Mode 函数中。

**BytePSGlobal::Hash_Mixed_Mode() **：这个 tensor 有 (2k(n-1)) / (n^2+kn-2k) 的几率分配给 k 个 non-colocate CPU servers 中任意一个， 有 1 - (2k(n-1)) / (n^2+kn-2k)  的几率分配给 n个 colocate CPU servers 中任意一个。这个分配方法和论文 section 4 的计算结果是等价地。