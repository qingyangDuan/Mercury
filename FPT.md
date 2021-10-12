# FPT (name in paper: Mercury)

## 1  modifications

file:  ps-lite/src/fpt_van.h;   ps-lite/include/ps/kv_app.h   

main class：**FPTVan** (based on ps-lite Class Van)

## 2  APIs

ps::KVWorker::  **FPTPush**    **FPTPull** （这两个函数会调用**FPTVan**的send函数完成push和pull。）

- FPTVan 只传输梯度数据。原始ps-lite的control message的传输仍由ZMQVan完成。

## 3  push pull callback defference

worker调用 push， pull 时会传入 cb ( callback ),  当下层库完成了相应的push pull工作后运行cb，即告知上层worker此任务已完成。

基于**FPTVan**的push和pull的完成判定与原始ps-lite有所不同，区别如下：

（注：上述ps-lite指mxnet1.5所依赖的版本。mxnet1.6所依赖的ps-lite已经有把push和pull融合在一起执行的操作：push_pull）



#### 3.1  原始 ps::KVWorker(based on ZMQVan)

**Push**: 所有worker基于Van的request执行push传输，即发送一个request给server（这个request携带push的梯度）。server接收到所有worker关于同一个key的push梯度并聚合后，才会发送response给所有worker（这个response没有任何数据）。worker接收到response之后，才会执行cb。

**Pull**：worker发送request给server。server接收到后，把聚合的梯度以response的形式发送给worker。worker把接受到的聚合梯度copy到相应的value的位置，然后执行cb。



#### 3.2  基于**FPTVan**的 ps::KVWorker

**FPTPush**: 所有worker发送相应key的梯度给server。server接收到所有worker关于同一个key的push梯度并聚合后，<font color=red>会自动发送聚合后的梯度给所有worker</font>。worker接收到聚合后的梯度并保存之后，才会执行cb。

**FPTPull**：worker把之前保存的聚合梯度copy到相应的value的位置，然后执行cb。

主要不同：FPTVan的 push操作 直接完成了所有传统意义上的 push传输、梯度聚合、pull回聚合梯度。所以FPTVan的pull操作只需完成一次本地数据copy。