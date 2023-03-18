# Overview

FastPS 即 patameter server load balancing 工作。



每个线程（W\S\H）只有一个全局PostOffice实例及其包含的一个或多个van，但可以有多个app，用 app_id区分。相同app_id 的app可互相通信。具体通信细节和通信范围见 ps-lite.md 的 2.2。FastPS 的顶层实现如下：

  - 原始ps-lite实现中，每个worker只创建一个KVWorker类型的app, 每个server只创建一个KVServer类型的app, 这些app之间通过 ZMQVan 传输 data msg， 即为梯度或参数。 scheduler不创建app。

- <font color=red>**FastPS 项目中，我们自定义了和 Load Balancing 相关的app： LBScheduler， LBWorker， LBServer 并在对应节点里创建，他们用相同的（LB_id ,  LB_id）标识。他们通过调用底层 ZMQVan 来传递 data msg ，从而实现 Aggregator的参数分配。** </font>

