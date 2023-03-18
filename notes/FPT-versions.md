# Github tocken for FPT-pslite

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

# Git commands

本地仓库管理更新：

- **git add [filename]**

- **git commit -m "xxxx"**

分支管理：

- **git branch**：查看分支命令
- **git branch [branchname]**：创建分支命令
- **git checkout [branchname]**：切换分支命令
- git merge：合并分支命令
- git branch -d [branchname]：删除分支命令

远程仓库管理：

- git remote add [reponame] [url]：添加远程仓库
- **git remote -v** ：查看当前的远程仓库
- **git push [reponame] [branchname]**：将当前分支push到远程仓库的branchname分支
- **git pull [reponame] [branchname]**：获取远程版本到本地remote文件夹，并合并到本地head文件夹。
- git fetch [reponame]：获取拴成版本到本地remote文件夹，不会自动合并
- git remote rm [reponame]：删除远程仓库

# 对外接口

MXNet 调用 ps-lite 实现传输。

FPT-ps-lite 沿用 ps-lite 的大部分接口，但有些许改变：

- 需要对 mxnet 的 train.py 及 kvstore_dist.h 等一系列文件增加一个函数。此函数作用为上层提前告诉 FPT-ps-lite，每一次迭代需要传输的 tensor的总数量。
- 在 kvstore_dist.h 把所调用的 ZPush() ， ZPull() 两个函数换为 FPTPush() 和 FPTPull()。 

接口和适配在不同版本中的现状：

- v1 及之前所有版本：FPT-ps-lite 接口一致，MXNet的对应适配一致。适配保存在老版MXNet文件夹中，其名字为  "(for FPTv1) incubator-mxnet-1.5" 。

- v2 及之后所有版本：FPT-ps-lite 接口和 MXNet 的对应适配都得到简化。

# papers

- Mercury：对应 v1 版本。

-  FastPS：对应v2版本。

# v0

最原始版本,包括v0.1 和 v0.2。每个worker节点创建N（number of servers）个接受线程，和N个发送线程。

每个server同理。

# v-epoll

基于版本 v0。

**改进内容:**

- 优化：用epoll 一个handle线程代替N个接收线程。发送仍为多线程。

**问题：**

- 用 EPOLLET模式 发现不能完全接收所有数据包。所以用EPOLLLT模式。
- 带宽跑不满 : 带宽利用率在80-100%间。
  - 解决：不执行setbuf。那么当worker和server数量都为1/2/3时，可跑满带宽；可当worker和server的数量都为4时，带宽利用率为80-100%之间。

# v1.1

基于版本： v0。

**改进内容:**

- 更新了FPT初始化建立TCP连接部分。worker或server调用 **FPTVan::Bind()**监听自己的监听tcp地址，其port比zmq监听地址的port大1。且只用一个listenfd监听即可生成多个connfd。
  - 缺点是：获得的一组 n个用于接收消息的connfd 并不能与 n个远程节点的id对上，即不知道哪个connfd对应于哪个远程节点。但是这不影响系统，因为这些只负责接收，只要发送端发送数据时告知id即可。
- 把 FPTVan 拆分成两个类 FPTWorkerVan 和 FPTServerVan。代码结构更清晰。

**功能欠缺：**

- 多节点第一个pull的值有问题，这是由于未能实现初始化时所有节点的init的barrier。以由 v1.2 解决。



## v1.1.1

- 改进了 priority_queues 的异步锁方式，即只用一个mutex。

## v1.1.2

- 在 FPTVan start() 中添加listen_thread.join()
- 删除了worker_init_queues_threads
- 实现了： 根据环境变量决定是否设置sndbuf 和 rcvbuf，以及设置的具体数值

**问题:**

-  设置setbuf 之后，带宽跑不满，无论数值是什么。不执行setbuf，反而可以跑满带宽。



# v1.2

基于版本 **v1.1**。

**改进内容:**

- Server端，只用一个发送线程代替 n个发送线程。这个发送线程对n个远程节点用轮询的方法发包。
- 实现了初始化所有节点都只拿到 第一个worker的数据， 即实现了其他节点的第一个pull。
- 完善 FPTWorkerVan 和 FPTServerVan 的 stop 函数。即关闭所有socket fds，且关闭所有收发和处理线程。
- 优化 tests 程序：

  - 在 test_kv_app 中实现了仿真trainer，可跑多次 iteration，每两个iteration间有barrier（即等待本次iteration的所有push pull全部完成，才进入下次迭代）。每个iteration内有多个不同优先级不同key的 tensor并发（这里实现为3个）。
- 实现一节点一个脚本（run_test.sh）即可跑多节点DML，且此脚本内自带net_monitor。

**问题:**

-  当tensor size太大时，程序会报错自动终止
  - 已解决。问题出在 自己定义的queue上，异步读写有处理冲突。
-     带宽跑不满：
   - 已解决：不设置setbuf 之后，带宽利用率跑满。

 **一些debug经验:**

- 自定义的queue的 读写操作 同时进行的 threadsafe问题
- 多节点多个push pull 消息到同一PS上，PS对这些消息的处理的 异步先后顺序问题。



# v2.1

基于版本 **v1.2**。

**改进内容：**

- 统一且简化上层的调用接口。
- 实现中心化 LBScheduler app： 收集 Worker 和 Server的带宽数据，并决定切分策略。

# v2.2

基于版本 **v2.1**。

**改进内容：**

-  实现worker和PS，都只用一个发送线程（实现FPTPool作为统一的待发送数据包缓存区）。
  - 接收仍为多线程，因此带宽可以跑满。可当worker和server的数量都为4时，带宽利用率为80-100%之间。

# v2.3

基于版本 **v2.2**。

**改进内容：**

-  实现worker和PS，都只用一个接受线程（epoll）实现。
  
  - CPU利用率大幅降低，且为恒定值，不随训练节点规模增长而增长。
  
  - worker和server数量都是 1~3 时，带宽可以跑满。**可当worker和server的数量都为4时，带宽利用率为80-100%之间。**

# v2.4

基于版本 **v2.3**。

**改进内容：**

- 实现 Worker 端和 Server 端都拥有 Aggregator 模块。
- 实现 FPTPool 根据各个flow的weight在pop时实现flow shaping。（LB app 在每个iteration开始前对FPTPool设置新的 weight）  
- 当分配 decision 发生变化时（每个新的iteration），FPTWorkerVan 和 FPTAggregator的本地 store_ 可以调整。
- 实现 scheduler 的decision有状态，如 probe_state。所有worker在执行 probe_state 的 decision 之前，会与所有server 同步，且告知它们各自所要接受的packet总数，以便所有servers设置用于探测带宽的参数。

# v2.5

基于版本 **v2.4**。

**改进内容：**

- 实现：epoll接收端的flow shaping。
- 实现：在probe状态时，禁用接收端的 flow shaping，以便更精确的测得servers的带宽。
- TODO：问题： **resnet50 训练时**，开启USE_PSLB：当decision在第3个term，从default decision（平分给所有aggregator）转换为一个只分给servers的decision时，会出现send_pool.Push() 错误，即产生了一个发送给别的worker中的aggregator的包。这是不应该出现的。 分析发现：训练时对同一个key调用了两次FPTPush和两次FPTPull后，iter\_count\_ 和 decision才更新。可能原因： worker端每num_TS 次 调用FPTPull后，执行更换decision。这个更换的时机可能不太对。

- TODO：问题：开启USE_PSLB：用 2 workers， 1 server 训练时，只能完成前三轮，之后就产生 `terminate called without an active exception` 的错误
- TODO: 当 LB_Scheduler 根据收集的bw信息计算决策时，bw里面有很多0或全是0怎么处理？