# ps-lite

## 1, main classes

![uml-final](ps-lite-uml.png)

- Postoffice是全局管理类，单例模式创建。主要用来配置当前node的一些信息，例如当前node是哪种类型(server,worker,scheduler)，nodeid是啥，以及worker/server 的rank 到 node id的转换。
- Van是负责通信的类，是Postoffice的成员。Van中std::unordered_map<int, void*> senders_保存了node_id到连接的映射。Van只是定义了接口，具体实现是依赖ZMQ实现的ZMQVan，Van类负责建立起节点之间的互相连接（例如Worker与Scheduler之间的连接），并且开启本地的receiving thread用来监听收到的message。
- Customer用来通信，跟踪request和response。每一个连接对应一个Customer实例，连接对方的id和Customer实例的id相同。
- SimpleApp是一个基类；提供了发送接收int型的head和string型的body消息，以及注册消息处理函数。它有2个派生类。
- KVServer是SimpleApp的派生类，用来保存key-values数据。里面的Process()被注册到Customer对象中，当Customer对象的receiving thread接受到消息时，就调用Process()对数据进行处理。
- KVWorker是SimpleApp的派生类，主要有Push()和Pull()，它们最后都会调用Send()函数，Send()对KVPairs进行切分，因为每个Server只保留一部分参数，因此切分后的SlicedKVpairs就会被发送给不同的Server。切分函数可以由用户自行重写，默认为DefaultSlicer，每个SlicedKVPairs被包装成Message对象，然后用van::send()发送。
- KVPairs封装了Key-Value结构，还包含了一个长度选项。
- SArray是Shared array，像智能指针一样共享数据，接口类似vector。
- Node封装了节点的信息，例如角色、ip、端口、是否是恢复节点。
- Control封装了控制信息，例如命令类型、目的节点、barrier_group的id、签名。
- Meta封装了元数据，发送者、接受者、时间戳、请求还是相应等。
- Message是要发送的信息，除了元数据外，还包括发送的数据。



## 2, W\S\H初始化及通信流程

以test_simple_app.cc为例，这是一个很简单的app，其它复杂的流程原理这个程序差不多，所以我们就说说这个程序是怎么运行的。worker(W)\Server(S)\Scheduler(H)三个角色所在的机器同时运行这个实例代码，本节介绍它们之间是怎么连接的。W\S\H代表这个脚本运行后各个角色后在不同角色程序内的处理流程。test_simple_app.cc主要代码为：

```c++
#include "ps/ps.h"
using namespace ps;

  Start(0); // (customer_id)
  SimpleApp app(0, 0); or  KVWorker worker(0,0) ;  or  KVServer server(0) ;// (app_id, customer_id) one App can have multiple customers 
  Finalize(0, true);


```

使用ps-lite主要是两步：(1) W\S\H 运行Start(),  (2) W\S创建对应自己角色的app（SimpleApp\KVWorker\KVServer）。

MXNet使用ps-lite的方法就是：a) W\S在KVStoreDist \ KVStoreDistServer的constructor中创建ps::KVWorker\ps::KVServer；b) W\S\H 运行ps::StartAsync()。ps::StartAsync 与ps::Start 的唯一区别是，StartAsync中，当Postoffice和 van的 start初始化完成之后，不会执行Barrier(7)，而需要再手动执行barrier。这里也说明了创建app和执行Start()可以不分先后。

### 2.1 Start初始化流程

W\S\H的Start函数会初始化PostOffice和Van，并初始化三种节点的互联。

- W\S\H：class Postoffice中：初始化static PostOffice和 它的 Van，全局都用同一个PostOffice --> Create(Van)用来做通信的收/发 --> 从环境变量中读入配置 --> 确定不同的角色。

- W\S\H：Start() --> PostOffice::Start() --> ZMQVan::Start() --> Van::Start()

  从 Van::Start() 开始， Van会执行 my_node_/Scheduler的初始化, 使worker和server都与scheduler绑定。 具体流程如下流程：

  - H ( Van::Start ) ：根据环境变量获得ip和port，并把van绑定到这个地址
  - W\S ( Van::Start ) ：根据环境变量获得interface，生成合适的ip和port，绑定van到这个地址。
  - W\S\H ( Van::Start ) ： <font color=red>在van中起一个Reciving的线程 ，这个线程会利用ZMQVan.RecvMsg()接受消息。这个线程的作用就是使得本节点可以接受其他节点传过来的控制信息。</font>
  - W\S (Van::Start) : 发送 ADD_NODE 消息给scheduler，从而连接Scheduler。
  - H ( Van::ProcessAddNodeCommand ) ：接受 ADD_NODE 消息
  - H ( Van::UpdataLocalID ) ：接受 ADD_NODE 消息， 把 server 和 worker 的节点信息保存在 meta* **nodes** 中。
  - H ( Van::ProcessAddNodeCommandAtScheduler ) ：收到所有的servers + worker 的 ADD_NODE信息后，便分配rank(即id) 。 把所有的nodes（包括H自己）的id、ip、port等信息打包在 meta* **nodes**中，然后发送 ADD_NODE消息给所有的 W\S, 当然这个消息中携带了 **nodes** 信息。 
  - W\S ( Van::ProcessAddNodeCommand ) : 收到 ADD_NODE 信息后，先在 Van::UpdataLocalID 更新自己的id为 H分配的数值。之后对消息中的 meta* **nodes** 信息包含的所有节点，都执行 ZMQVan::Connect(node)，即连接到了所有的其他节点。 <font color=blue>当然ZMQVan中会避免重复连接同一个节点（因此此前 W\S 已经连接过了H）；也会避免 server连接server、worker连接worker，因为这是无意义的。</font>
  - W\S\H ( PostOffice::Start ) : Van::Start()结束后，PostOffice::Start() 执行 Barrier()。 这个函数会向scheduler发送 BARRIER 消息并阻塞主线程。scheduler的 Van::Receiving() 线程收到所有的 node 的 barrier msg，即表示所有节点都完成了上述的初始化，然后scheduler解除自己及所有node的barrier，主线程恢复。 <font color=blue> BARRIER消息中的 msg.meta.request 为 true 代表一个普通的 BARRIER消息， 为false则代表一个解除当前 BARRIER的消息。</font>

### 2.2 创建app，并绑定app的customer到全局PostOffice

  - W\S：初始化app ：在app内部创建New Customer（用 app_id 和 customer_id 标识），并绑定app::Process作为Customer的recv_handle_，调用PostOffice::AddCustomer将当前Customer注册到全局PostOffice，从而可以接受van上传的消息 --> <font color=red>Customer起一个Receiving线程</font>

一个线程只有一个全局PostOffice实例及其包含的van，可以有多个app。每个app内部有一个customer实例，其主要用于app接收消息。具体如下：

- customer实例在初始化时被注册到全局PostOffice实例中，用（app_id, customer_id）区分。customer实例会起一个Receiving线程，用于接受van传递给它的msg。van接收到msg后根据（app_id, customer_id) 在全局PostOffice实例中寻找对应的customer，然后把msg给这个customer的Receiving线程处理。

每个app调用全局PostOffice实例的van发送消息，发送时注明app_id和customer_id，以便接收方的van能够找到正确的destination app。 每个app通过自身的customer实例接收消息。



### 2.3 消息处理流程

每个节点都监听了本地一个端口，该连接的节点在启动时已经连接。

- `Van::Receiving()`函数是单独一个线程来接收消息。它会接收到两种消息： controlMsg和 dataMsg。前者是Van内部用于控制启停及各节点互联初始化的消息，一般没有数据（这些消息主要应用在 Start()函数执行阶段, 如 TERMINATE,  ADD_NODE, BARRIER,  HEATBEAT等等特殊命令）；后者是外部通过调用app的函数传递的消息，一般包含数据（如调用KVWorker，KVServer的push、pull传递数据）。 接收到消息后，如果是controlMsg，则根据不同 msg.meta.control.cmd 执行不同动作，在Van类内处理（例如`Control::ADD_NODE`就是添加节点）；如果是 DataMsg，会根据消息的app_id 和 customer_id 找到指定的customer对象，然后将消息传递给该对象的`Customer::Accept`函数。

以下假设创建的app是SimpleApp：

- `Customer::Accept()`函数将消息添加到一个队列`recv_queue_`；`Customer::Receiving()`是一个线程在运行，从队列取消息处理；处理过程中会使用函数对象`recv_handle_`处理消息，这个函数对象是`SimpleApp::Process`函数。
- `SimpleApp::Process`根据是消息类型（请求or响应），调用用户注册的函数来处理消息，`request_handle_`、`response_handle_`分别处理请求和响应。

以上只是原始SimpleApp基类的处理流程，如果创建的app是其派生类KVWorker， KVServer， 那么设置的处理函数句柄`recv_handle_` `request_handle_` `response_handle_`会有所不同。

### 2.4 消息处理流程KVWorker/KVServer版

KVWorker和KVServer都继承自SimpleApp类。

1）对于worker来说，其注册的`recv_handle_`是`KVWorker::Process()`函数。因为worker的recv thread接受到的消息主要是从server处pull下来的KV对，因此该`Process()`主要是接收message中的KV对；

2）而对于Server来说，其注册的`recv_handle_`是`KVServer::Process()`函数。因此server接受的是worker们push上来的KV对，需要对其进行处理，因此该`Process()`调用用户通过`KVServer::set_request_handle()`传入的函数句柄来处理消息。

### 2.5 Customer 记录消息控制同步异步依赖

每个customer对象都拥有一个`tracker_`(`std::vector<std::pair<int, int>>`类型)用来记录每个请求发送和返回的数量。
`tracker_`的下标即为请求的timestamp，`tracker_[t].first`是该请求发送给了多少节点，`tracker[t]_.second`是该请求收到了多少节点的回复。`customer::Wait()`就是一直阻塞直到`tracker_[t].first == tracker[t].second`，用这个来控制同步异步依赖。

Customer::Accept函数处理`Van::Receiving()` 传递的 dataMsg时，同时该message对应的请求（设为req）则`tracker_[req.timestamp].second++`。



### 2.6 Customer处理信息流程

Customer处理普通信息流程如下：

- H：app->requst() --> 放这个请求入到tracker_中 --> send(msg) --> app->wait() [等待收回发的信息]
- W/S：收到信息后放到recv_queue_中
- W/S：Customer的Reciving收到信息 --> call recv_handle_ --> process(recv)[处理信息] --> response_hadle_(recv) --> ReqHandle() --> response()[回发信息]
- H：收到回发的信息 --> 放入到recv_queue_中处理 --> 在Customer中的Reciving中处理
- H：当tracker_.first == tracker_.second时，释放app->wait()



## 3 ,一些实现细节

###  3.1 位运算表示node和node group

因为有时请求要发送给多个节点，所以ps-lite用了一个map来存储**每个id对应的实际的node节点**。

其中id：1,2,4分别表示Scheduler, ServerGroup, WorkerGroup。
这样只需要将请求的目标节点的id 设为4，便意味着将该请求发送到所有的worker node。

除此之外，如果某worker想要向所有的server和scheduler同时发送请求，则只需要将目标node_id设为3即可。因为 3=2+1。

这正是为什么会选择1,2,4的原因。因此1-7内任意一个数字都代表的是Scheduler/ServerGroup/WorkerGroup的某一种组合。

1-7的id表示的是node group，而后续的id（即8，9，10，······）则表示单个的node。
其中8，10，12表示 worker0，worker1，worker2（即 2n+8）； 9，11，13 表示server0，server1，server2（即2n+9）。

如此来说，对于每一个新节点，需要将其对应多个id上。例如对于worker2来说，需要将它与4,4+1,4+2,4+1+2,12这4个id相对应。

### 3.2 KVPairs把keys数组和values数组分开

KVPairs的数据结构并非按照 `vector<pair<key, vector<values>>>`，而是按照`vector<key>`, `vector<values>`来组成。
这是因为，对于worker来说，它所拥有的部分数据集train data通常都是不变的，那这些数据集所引用的keys通常也是不变的。
这样的话，worker和server之间互相通信的时候，就可以不发送vector，仅发送vector了，可以降低一部分网络带宽。

### 3.3 工作类SimpleApp，KVWorker，KVServer的初始化和api

SimpleApp 和 Customer类中的 id：
   * app_id： the globally unique id indicating the application the postoffice serving for

   * customer_id： the locally unique id indicating the customer of a postoffice

**SimpleApp(app_id, custom_id)**

- 初始化时： 新建一个Custom对象初始化obj_ 成员；分别用传入构造函数的参数初始化app_id_, custom_id_, recv_handle成员；调用PostOffice::AddCustomer将当前Customer注册到PostOffice；新起一个Receiving线程recv_thread_；
- set_request_handle，set_response_handle：设置成员request_handle_, response_handle_。在客户端调用SimpleApp::Process时，根据message.meta中的指示变量判断是request还是response，调用相应handle处理；

**KVServer(app_id)**

- 初始化时：新建一个Customer对象初始化obj_成员，用KVServer::Process传入Customer构造函数，对于Server来说，app_id=custom_id=server's id；
- set_request_handle，在调用KVServer::Process时，该函数使用request_handle处理message

**KVWorker(app_id, custom_id)**

- 初始化时：用默认的KVWorker::DefaultSlicer绑定slicer_成员；新建一个Customer对象初始化obj_成员，不传入handle参数；

- set_slicer：设置slicer_成员，该函数在调用Send函数时，将KVPairs按照每个server的Range切片；
- **Pull**(key_vector, val_vector, option: len_vector, cmd, callback)：根据key_vector从Server上拉取val_vector，返回timestamp，该函数不阻塞，可用worker.Wait(timestamp)等待；
- ZPull同理Pull，不过在调用内部Pull_函数时，不会copy一个key_vector，所以需要保证在ZPull完成前，调用者没有改变key_vector；
- **Push**(key_vector, val_vector, optional: len_vector, cmd, callback) 以及ZPush：

   1. 由obj_成员准备一个送到ServerGroup的request返回stamp；
   2. 设置好对应timestamp的callback；
   3. 使用传入的参数构造KVPair对象，调用Send送出该对象；



### 3.4 Barrier

- 初始化时：所有节点在 PostOffice::Start() 结尾时设置BARRIER，即发送 barrier msg（group为7，request为true）给Scheduler， 并阻塞当前主线程。Scheduler每接受到一个barrier msg后，在 Van::ProcessBarrierCommand 中增加 barrier_count_[group]。接收完所有barrier msg 后，Scheduler 给所有节点发送  取消barrier msg（request为False）。
- Van::Receiving() 线程中的 Van::ProcessBarrierCommand 中收到的 msg 处理逻辑：
	- 如果 msg->meta.request 为 true，则对 barrier_count_[msg->meta.control.barrier_group] 加1 。如果这个数值达到了 GetNodeIDs(group).size() ，则对这个 group 中的所有node 发送 取消barrier（request 为 false）
	- 如果 msg->meta.request 为 false，则由 Postoffice::Get()->Manage(*msg) 执行 barrier_cond_.notify_all()  ，从而接触当前node的barrier。
	- Postoffice::Barrier() 中会设置 barrier_cond_.wait(), 从而阻塞当前线程。

### 3.5 id rank

Every node's id is unique, scheduler's id is always 1.

- **scheduler id is 1**
- servers rank：0、1、2、3......; **server node.rank to id**: id = rank * 2 + 8
- workers rank：0、1、2、3......; **worker node.rank to id**: id = rank * 2 + 9
- **node.id to rank**:  rank = std::max((id - 8) / 2, 0)

用一个 id 表示一组节点：

-  kScheduler = 1;   kServerGroup = 2;   kWorkerGroup = 4;
- 例如：发送给id=6表示------发送给所有servers+workers

​      

