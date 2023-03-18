# overview

- 自定义两种 FPTVan：**FPTWorkerVan**   **FPTServerVan** (继承自 Van)

- 原始 ps-lite 的每个线程（W\S\H）的全局PostOffice实例只有一个 ZMQVan，本工作在 W\S 节点另建 FPTVan，且在ZMQVan建立通信连接的同时也完成任一 W 和任一 S 的通信连接。
  -  FPTVan负责带优先级的梯度数据传输。
  - 原始ps-lite的control message的传输仍由ZMQVan完成。

# APIs

ps::KVWorker:: **SetPriorityNum()**  **FPTPush()**    **FPTPull()** （后两个函数会调用**FPTVan**的send函数完成push和pull。）

# push pull cb defference

worker调用 push / pull 时会传入 cb ( callback ),  当下层库完成了相应的push / pull工作后运行cb，即告知上层worker此任务已完成。

基于**FPTVan**的push和pull与原始ps-lite有所不同，区别如下：

（注：上述ps-lite指mxnet1.5所依赖的版本。mxnet1.6所依赖的ps-lite已经有把push和pull融合在一起执行的操作：push_pull）

## 1) 原始 ps::KVWorker(based on ZMQVan)

**Push**: 所有worker基于Van的request执行push传输，即发送一个request给server（这个request携带push的梯度）。server接收到所有worker关于同一个key的push梯度并聚合后，才会发送response给所有worker（这个response没有任何数据）。worker接收到response之后，才会执行cb。

**Pull**：worker发送request给server。server接收到后，把聚合的梯度以response的形式发送给worker。worker把接受到的聚合梯度copy到相应的value的位置，然后执行cb。



## 2) 基于**FPTVan**的 ps::KVWorker

**FPTPush**: 所有worker发送相应key的梯度给server。server接收到所有worker关于同一个key的push梯度并聚合后，<font color=red>会自动发送聚合后的梯度给所有worker</font>。worker接收到聚合后的梯度并保存之后，才会执行cb。即执行cb时，实际上pull的传输已完成，本节点已经拿到聚合后的梯度。

**FPTPull**：worker把保存的聚合后的梯度copy到相应的value的位置，然后执行cb。

主要不同：FPTVan的 push操作 直接完成了所有传统意义上的 push传输、梯度聚合、pull回聚合梯度。所以FPTVan的pull操作只需完成一次本地数据copy。

# FPT初始化建立TCP连接

Scheduler的FPT实例不工作，只需建立所有Worker与所有Server之间的TCP连接即可。这与ZMQVan所建立的通信连接拓扑是一样的。因此可模仿 ZMQVan 的建立连接流程。

## Start初始化流程with FPT

下面步骤为原始ps-lite的 ZMQ的初始化建立连接流程，只有红色的部分为FPT加入的初始化步骤。



W\S\H的Start函数会初始化PostOffice和Van，并初始化三种节点的互联。

- W\S\H：class Postoffice中：初始化static PostOffice和 它的 Van实例（ZMQ_Van），<font color=red>初始化 FPT_Van。</font>全局都用同一个PostOffice --> Create(Van)用来做通信的收/发 --> 从环境变量中读入配置 --> 确定不同的角色。

- W\S\H：Start() --> PostOffice::Start() --> ZMQVan::Start() --> Van::Start()

  从 Van::Start() 开始， Van会执行 my_node_/Scheduler的初始化, 使worker和server都与scheduler绑定。 具体流程如下流程：

  - W\S\H ( Van::Start ) ：H：获取环境变量（root ip和root port）作为自己的监听地址。W\S ：获取root地址用于连接H，也生成自己的监听地址（ip和port）。
  - W\S\H ( Van::Start ) ：调用**ZMQVan::Bind()监听** 自己的监听地址，调用 **ZMQVan::Connect(scheduler)**与scheduler建立zmq连接。
  - W\S\H ( Van::Start ) ：<font color=blue>在van中起一个Reciving的线程 ，这个线程会利用ZMQVan.RecvMsg()接受消息。这个线程的作用就是使得本节点可以接受其他节点传过来的控制信息。</font>当前时刻，只有H与所有其他节点有连接，W与S之间无连接。
  - W\S (Van::Start) : 发送 ADD_NODE 消息给scheduler，从而把自己的本地监听地址发送给Scheduler。
  - H ( Van::ProcessAddNodeCommand ) ：接受 ADD_NODE 消息
  - H ( Van::UpdataLocalID ) ：接受 ADD_NODE 消息， 把 server 和 worker 的节点信息（本地监听地址）保存在 meta*  nodes 中。
  - H ( Van::ProcessAddNodeCommandAtScheduler ) ：收到所有的servers + worker 的 ADD_NODE信息后，便分配rank(即id) 。 把所有的nodes（包括H自己）的id、ip、port等信息打包在 meta* nodes中，然后发送 ADD_NODE消息给所有的 W\S, 当然这个消息中携带了 所有nodes 的信息。 
  - W\S ( Van::ProcessAddNodeCommand ) : 收到 ADD_NODE 信息后，先在 Van::UpdataLocalID 更新自己的id为 H分配的数值。之后对消息中的 meta* nodes 信息包含的所有节点，都调用 **ZMQVan::Connect(node)**，即连接到了所有的其他节点。当然ZMQVan中会避免重复连接同一个节点（因此此前 W\S 已经连接过了H）；也会避免 server连接server、worker连接worker，因为这是无意义的。即这一步其实是W与S互相建立zmq连接。 <font color=red>对消息中的 meta* nodes 信息包含的所有节点，都调用 **FPTVan::Connect(node)**，使得W与S 间建立tcp连接。</font>
  - W\S\H ( Van::Start )  Van:::Start 等待所有的连接建立完成之后才会退出。
  -  <font color=red>W\S( FPT_Van::Start )：执行**FPTVan::Start()**, 创建发送接收线程。</font>
  - W\S\H ( PostOffice::Start ) : Van::Start()结束后，PostOffice::Start() 执行 Barrier()。 这个函数会向scheduler发送 BARRIER 消息并阻塞主线程。scheduler的 Van::Receiving() 线程收到所有的 node 的 barrier msg，即表示所有节点都完成了上述的初始化，然后scheduler解除自己及所有node的barrier，主线程恢复。 <font color=blue> BARRIER消息中的 msg.meta.request 为 true 代表一个普通的 BARRIER消息， 为false则代表一个解除当前 BARRIER的消息。</font>



## Start初始化流程with FPT（ps-lite-epoll 版本）

接收端用epoll优化，即只用一个epoll handle线程处理所有接收的tcp连接。基于epoll  优化TCP连接的FPT版本：



W\S\H的Start函数会初始化PostOffice和Van，并初始化三种节点的互联。

- W\S\H：class Postoffice中：初始化static PostOffice和 它的 Van实例（ZMQ_Van），<font color=red>初始化 FPT_Van。</font>全局都用同一个PostOffice --> Create(Van)用来做通信的收/发 --> 从环境变量中读入配置 --> 确定不同的角色。

- W\S\H：Start() --> PostOffice::Start() --> ZMQVan::Start() --> Van::Start()

  从 Van::Start() 开始， Van会执行 my_node_/Scheduler的初始化, 使worker和server都与scheduler绑定。 具体流程如下流程：

  - W\S\H ( Van::Start ) ：H：获取环境变量（root ip和root port）作为自己的监听地址。W\S ：获取root地址用于连接H，也生成自己的监听地址（ip和port）。
  - W\S\H ( Van::Start ) ：调用**ZMQVan::Bind()监听** 自己的监听地址，<font color=red>调用 **FPTVan::Bind()**监听自己的监听tcp地址，其port比zmq监听地址的port大1。Bind会创建epoll event handle线程，即接收包线程。</font>调用 **ZMQVan::Connect(scheduler)**与scheduler建立zmq连接。
  - W\S\H ( Van::Start ) ：<font color=blue>在van中起一个Reciving的线程 ，这个线程会利用ZMQVan.RecvMsg()接受消息。这个线程的作用就是使得本节点可以接受其他节点传过来的控制信息。</font>当前时刻，只有H与所有其他节点有连接，W与S之间无连接。
  - W\S (Van::Start) : 发送 ADD_NODE 消息给scheduler，从而把自己的本地监听地址发送给Scheduler。
  - H ( Van::ProcessAddNodeCommand ) ：接受 ADD_NODE 消息
  - H ( Van::UpdataLocalID ) ：接受 ADD_NODE 消息， 把 server 和 worker 的节点信息（本地监听地址）保存在 meta*  nodes 中。
  - H ( Van::ProcessAddNodeCommandAtScheduler ) ：收到所有的servers + worker 的 ADD_NODE信息后，便分配rank(即id) 。 把所有的nodes（包括H自己）的id、ip、port等信息打包在 meta* nodes中，然后发送 ADD_NODE消息给所有的 W\S, 当然这个消息中携带了 所有nodes 的信息。 
  - W\S ( Van::ProcessAddNodeCommand ) : 收到 ADD_NODE 信息后，先在 Van::UpdataLocalID 更新自己的id为 H分配的数值。之后对消息中的 meta* nodes 信息包含的所有节点，都调用 **ZMQVan::Connect(node)**，即连接到了所有的其他节点。 <font color=blue>当然ZMQVan中会避免重复连接同一个节点（因此此前 W\S 已经连接过了H）；也会避免 server连接server、worker连接worker，因为这是无意义的。</font>即这一步其实是W与S互相建立zmq连接。 <font color=red>对消息中的 meta* nodes 信息包含的所有节点，都调用 **FPTVan::Connect(node)**，使得W与S 间建立tcp连接。</font>
  - W\S\H ( Van::Start )  Van:::Start 结束。
  -  <font color=red>W\S( FPT_Van::Start )：执行**FPTVan::Start()**, 创建发送线程、接收包处理线程。</font>
  - W\S\H ( PostOffice::Start ) : Van::Start()结束后，PostOffice::Start() 执行 Barrier()。 这个函数会向scheduler发送 BARRIER 消息并阻塞主线程。scheduler的 Van::Receiving() 线程收到所有的 node 的 barrier msg，即表示所有节点都完成了上述的初始化，然后scheduler解除自己及所有node的barrier，主线程恢复。 <font color=blue> BARRIER消息中的 msg.meta.request 为 true 代表一个普通的 BARRIER消息， 为false则代表一个解除当前 BARRIER的消息。</font>
