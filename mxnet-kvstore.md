

# Mxnet kvstore(python) to ps-lite(c++) api

mxnet-1.5

## 1 python Classes 

### Module [fit(): "symbolic" training]

file: python/mxnet/module/base_module.py     python/mxnet/module/module.py

members:

methods:

- init_params() : something with cache and devices

- init_optimizer() :  create **self._optimizer** with `rescale_grad = 1/batch_size`;  这个函数会通过用户指定的参数来调用python/mxnet/**model.py**中的**`_create_kvstore`**来初始化`kvstore`以及`update_kv_store`这两个变量。其中`kvstore`是`KVStore`类的一个实例化对象，而`update_on_kvstore`是一个布尔型变量，用来判断是否在ps端更新参数。换句话说，如果该变量为True，那么模型参数的更新发生在ps端；否则，模型参数的更新发生在worker端，ps端只做梯度的聚合操作（这种情况下，paramerter server是不是就变成了gradient server？）。然而，只有在同步训练模式下，我们才能设置`update_on_kvstore=False`，异步训练并不支持在worker端更新参数。在`update_kv_store=True`的情况下，我们需要告诉ps端训练过程中使用的优化器是什么，因此要调用`kvstore.set_optimizer`把优化器从worker端传给ps端

- fit() ( in Class **BaseModule** ) :    运行 self.init_optimizer()。运行forward，backward，update。

- forward(data_batch) :

- backward() :

- update()  :
	
	```python
	        _update_params_on_kvstore(self._exec_group.param_arrays,
	                                  self._exec_group.grad_arrays,
	                                  self._kvstore, self._exec_group.param_names)
	```
	
	- _update_params_on_kvstore(param_arrays, grad_arrays, kvstore, param_names) (in python/mxnet/**model.py**) : 
	
	```python
	for index, pair in enumerate(zip(param_arrays, grad_arrays)):
	        arg_list, grad_list = pair
	        if grad_list[0] is None:
	            continue
	        name = param_names[index]
	        # push gradient, priority is negative index
	        kvstore.push(name, grad_list, priority=-index)
	        # pull back the weights
	        kvstore.pull(name, arg_list, priority=-index)
	```
	



### Optimizer

file: python/mxnet/optimizer/optimizer.py

members:

- **self.rescale_grad** = 1.0/batch_size：  *Multiply the gradient with `rescale_grad` before updating.* 

  


### KVStore

file: python/mxnet/kvstore.py

```python
    """
    name : {'local', 'device', 'nccl', 'dist_sync', 'dist_device_sync', 'dist_async'}

    single machine:

    ``local``: Copies all gradients to CPU memory and updates weights there.

    ``device``: Aggregates gradients and updates weights on GPUs. With this setting,
    the KVStore also attempts to use GPU peer-to-peer communication,
    potentially accelerating the communication.

    Distributed training:

    ``dist_sync``: Behaves similarly to ``local`` but with one major difference.
    With ``dist_sync``, batch-size now means the batch size used on each machine.
    So if there are ``n`` machines and we use batch size ``b``,
    then ``dist_sync`` behaves like ``local`` with batch size ``n * b``.

    ``dist_device_sync``: Identical to ``dist_sync`` with the difference similar
    to ``device`` vs ``local``.

    ``dist_async``: Performs asynchronous updates.
    The weights are updated whenever gradients are received from any machine.
    No two updates happen on the same weight at the same time. However, the order is not
    guaranteed.
    """
```



members:

methods:

- **init()** :

  Initializes a single or a sequence of key-value pairs into the store. For each key, one must `init` it before calling `push` or `pull`.  When multiple workers invoke `init` for the same key, only the value supplied by worker with rank `0` is used. <font color=red>This function returns  after data has been initialized successfully</font>.

  ```python
      def init(self, key, value):
          """ 
          Parameters
          ----------
          key : str, int, or sequence of str or int
              The keys.
          value : NDArray, RowSparseNDArray or sequence of NDArray or RowSparseNDArray
              Values corresponding to the keys.
  
          """
          ckeys, cvals, use_str_keys = _ctype_key_value(key, value)
          if use_str_keys:
              check_call(_LIB.MXKVStoreInitEx(self.handle, mx_uint(len(ckeys)), ckeys, cvals))
          else:
              check_call(_LIB.MXKVStoreInit(self.handle, mx_uint(len(ckeys)), ckeys, cvals))
  ```

  

- **push()** : 

  This function returns immediately after adding an operator to the engine. The actual operation is executed asynchronously. <font color=red>If there are consecutive  pushes to the same key, there is no guarantee on the serialization of pushes. The execution of a push does not guarantee that all previous pushes are finished.</font> There is no synchronization between workers. One can use ``_barrier()`` to sync all workers.  

  - **const_vars: in_tensor 即 value  ;   mutable_vars: nothing **
  
  - 这里的参数 **value** 和下面pull函数的 参数 **out** 都是 **NDArray** 实例或其list。NDArray.handle 是 **NDArrayHandle** 实例，其中`NDArrayHandle = ctypes.c_void_p` 。在python NDArray实例创建时，其 handle已经指向了_LIB 中的一个 c++ NDArray实例，代表python中的NDArray实例在c++中的映射。
  ```python
          """Parameters：
      	key : str, int, or sequence of str or in Keys.
  
          value : NDArray, RowSparseNDArray, list of NDArray or RowSparseNDArray,
                  or list of list of NDArray or RowSparseNDArray
              Values corresponding to the keys.
  
          priority : int, optional
              The priority of the push operation.
              Higher priority push operations are likely to be executed before
              other push actions.
              """
      ckeys, cvals, use_str_keys = _ctype_key_value(key, value)
      if use_str_keys:
          check_call(_LIB.MXKVStorePushEx(
              self.handle, mx_uint(len(ckeys)), ckeys, cvals, ctypes.c_int(priority)))
      else:
          check_call(_LIB.MXKVStorePush(
              self.handle, mx_uint(len(ckeys)), ckeys, cvals, ctypes.c_int(priority)))
  ```

- **pull()** :

  This function returns immediately after adding an operator to the engine. <font color=red>Subsequent attempts to read from the `out` variable will be blocked until the pull operation completes. `pull` is executed asynchronously after all previous `pull` calls and only the last

   `push` call for the same input key(s) are finished</font>.  The returned values are guaranteed to be the latest values in the store.
  
  - **const_vars: nothing  ;   mutable_vars: out_tensor**
  
  ```python
          """Parameters
          ----------
          key : str, int, or sequence of str or int
              Keys.
  
          out: NDArray or list of NDArray or list of list of NDArray
              Values corresponding to the keys.
  
          priority : int, optional
              The priority of the pull operation.
              Higher priority pull operations are likely to be executed before
              other pull actions.
      	"""
      assert(out is not None)
      ckeys, cvals, use_str_keys = _ctype_key_value(key, out)
      if use_str_keys:
          check_call(_LIB.MXKVStorePullWithSparseEx(self.handle, mx_uint(len(ckeys)), ckeys,
                                                    cvals, ctypes.c_int(priority),
                                                    ctypes.c_bool(ignore_sparse)))
      else:
          check_call(_LIB.MXKVStorePullWithSparse(self.handle, mx_uint(len(ckeys)), ckeys,
                                                  cvals, ctypes.c_int(priority),
                                                  ctypes.c_bool(ignore_sparse)))
  ```



### KVStoreServer

- file: python/mxnet/kvstore_server.py

methods:

- \__init__ :

  ```python
      def __init__(self, kvstore):
          """Initialize a new KVStoreServer.
  
          Parameters
          ----------
          kvstore : KVStore
          """
          self.kvstore = kvstore
          self.handle = kvstore.handle
          self.init_logginig = False
  ```

  

- run :

  ```python
      def run(self):
          """Run the server, whose behavior is like.
          """
          _ctrl_proto = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
          check_call(_LIB.MXKVStoreRunServer(self.handle, _ctrl_proto(self._controller()), None))
  ```

  




## 2 how to use C++ library in python?

###  _LIB (python 实例) 

- 创建过程（in python/mxnet/base.py）(与 **libmxnet.so**链接)

  ```python
      def _load_lib():
          """Load library by searching possible path."""
          lib_path = libinfo.find_lib_path()
          lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_LOCAL)
          # lib_path[0]: path of libmxnet.so
  
          # DMatrix functions
          lib.MXGetLastError.restype = ctypes.c_char_p
          return lib
  
  
      # version number
      __version__ = libinfo.__version__
      # library instance of mxnet
      _LIB = _load_lib()
  ```

  

- **_LIB** 是python 中的c++库实例。用它来调用c++ 库函数。

- **linmxnet.so** 是被mxnet 的 cmake创建的共享库，几乎包含所有的 .cc  .h 文件, 包括依赖的第三方c++库，如ps-lite

  

### c_api.cc (mxnet c++ 文件)

这个c++文件包含 **_LIB** 实例所调用的c++库函数定义

- file location：src/c_api/c_api.cc

functions：

- **MXKVStoreInit()**

  ```python
  int MXKVStoreInit(KVStoreHandle handle,
                    mx_uint num,
                    const int* keys,
                    NDArrayHandle* vals) {
    API_BEGIN();
    std::vector<int> v_keys(num);
    std::vector<NDArray> v_vals(num);
    for (mx_uint i = 0; i < num; ++i) {
      v_keys[i] = keys[i];
      v_vals[i] = *static_cast<NDArray*>(vals[i]);
    }
    static_cast<KVStore*>(handle)->Init(v_keys, v_vals);
    API_END();
  }
  ```

  

- **MXKVStorePush()**

  ```c++
  int MXKVStorePush(KVStoreHandle handle,
                    mx_uint num,
                  const int* keys,
                    NDArrayHandle* vals,
                    int priority) {
    API_BEGIN();
    std::vector<int> v_keys(num);
    std::vector<NDArray> v_vals(num);
    for (mx_uint i = 0; i < num; ++i) {
      v_keys[i] = keys[i];
      v_vals[i] = *static_cast<NDArray*>(vals[i]);
    }
    static_cast<KVStore*>(handle)->Push(v_keys, v_vals, priority);
    API_END();
  }
  ```

### Handle机制（映射 python类实例和c++类实例）

重要的python类在创建实例初始化时，有一个 member 叫做 **self.handle**。它通常是 **ctypes.c_void_p** ，即一个在python中表示 c++指针的对象。初始化时通常会通过 _LIB 的某些函数把这个handle 指向 c++ 的某个类实例，从而把 python的类实例映射到c++的类实例。

已知的有这种机制的 python类： KVStore ，NDArray。

下面是 KVStore 初始化时设置handle的过程：

- python类KVStore实例创建函数如下：

  ```python
  #python code：in python/mxnet/kvstore.py
  # usage example： kv_store = mxnet.kv.create('dist_sync')
  KVStoreHandle = ctypes.c_void_p ## void * 类型指针
  def create(name='local'):
  
      if not isinstance(name, string_types):
          raise TypeError('name must be a string')
      handle = KVStoreHandle()
      check_call(_LIB.MXKVStoreCreate(c_str(name),
                                      ctypes.byref(handle)))
      kv = KVStore(handle)
      set_kvstore_handle(kv.handle)
      return kv
  ```
  
  - 上面新建了一个handle 给KVStore实例，并调用\_LIB.MXKVStoreCreate 函数设置这个handle 在c++中的指向。\_LIB.MXKVStoreCreate 如下：
  	```c++
  // c++ code : in src/c_api/c_api.cc
  typedef void *KVStoreHandle;
    int MXKVStoreCreate(const char *type,
                      KVStoreHandle *out) {
    API_BEGIN();
    *out = KVStore::Create(type);
  API_END();
  	}
		```
  
  	
  	- 其中上面的 Create 函数是 (如果使用'dist_*', 返回的就是 **KVStoreDist** 类指针) ：
  	
  	  ```c++
  	  // c++ code in src/kvstore/kvstore.cc
  	  	  KVStore* KVStore::Create(const char *type_name) {
  	  	    std::string tname = type_name;
  	  	    std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
  	  	    KVStore* kv = nullptr;
  	  	    bool use_device_comm = false;
  	  	    auto has = [tname](const std::string& pattern) {
  	  	      return tname.find(pattern) != std::string::npos;
  	  	    };
  	  	    if (has("device")) {
  	  	      use_device_comm = true;
  	  	    }
  	  	  
  	  	    if (has("dist")) {
  	  	  #if MXNET_USE_DIST_KVSTORE
  	  	      kv = new kvstore::KVStoreDist(use_device_comm);
  	  	      if (!has("_async") && kv->IsWorkerNode() && kv->get_rank() == 0) {
  	  	        // configure the server to be the sync mode
  	  	        kv->SendCommandToServers(static_cast<int>(kvstore::CommandType::kSyncMode), "");
  	  	      }
  	  	  #else
  	  	      LOG(FATAL) << "compile with USE_DIST_KVSTORE=1 to use " << tname;
  	  	      return nullptr;
  	  	  #endif  // MXNET_USE_DIST_KVSTORE
  	  	    } else {
  	  	      if (has("nccl")) {
  	  	  #if MXNET_USE_NCCL
  	  	        kv = new kvstore::KVStoreNCCL();
  	  	  #else
  	  	        LOG(FATAL) << "compile with USE_NCCL=1 to use " << tname;
  	  	        return nullptr;
  	  	  #endif
  	  	      } else {
  	  	        kv =  new kvstore::KVStoreLocal(use_device_comm);
  	  	      }
  	  	    }
  	  	    kv->type_ = tname;
  	  	    return kv;
  	  	  }
  	  ```
  	
  	  ​	  
  	

## 3 mxnet c++  KVStore classes

files： src/kvstore/*.h  & *.cc ;  include/mxnet/kvstore.h

**KVStoreDist**  **KVStoreDistServer** 依赖于**ps-lite** 的类和库

classes：

### KVStoreDist （c++ class）

KVStoreDist 继承自 KVStoreLocal ； 而 KVStoreLocal 继承自 KVStore。

当我们使用`dist-*`去create KVStore的时候，就会使用到类`KVStoreDist`。`KVStoreDist`分两个主要部分，一个是worker，一个是server。 如果该节点是**worker**，首先会创建一个`ps_worker_ = new ps::KVWorker<char>(0, new_customer_id);`这个`ps::KVWorker`将在**ps-lite**部分具体解析，它是主要的完成`push`和`pull`操作的部分。

members:

```c++
  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<char>* ps_worker_;
  /**
   * \brief the server handle
   */
  KVStoreDistServer* server_;

  /// reducer and broadcaster
  Comm* comm_
      
    /**
   * \brief buffer for non-compressed data.
   *  map of key and NDArray
   */
  std::unordered_map<int, NDArray> comm_buf_;

  // environment variable for server load balancing
  bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);

  /**
   * \brief cache all key partitions
   *
   * `ps_kv_` is used for pushes and pulls without gradient compression
   * `compr_ps_kv_` is used for gradient compression. It contains different
   * pskv for push and pull because sizes would be different in both cases.
   * Note: `ps_kv_[k]` for some key k may not be the same as `compr_ps_kv_[k].pull`
   * This is because sharding may cause slightly different divisions when size is
   * not perfectly divisible.
   */
  std::unordered_map<int, PSKV> ps_kv_;
  std::unordered_map<int, ComprPSKV> compr_ps_kv_;

  /**
   * \brief struct for ps keys and lens
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
  };
```

methods：

#### constructor 实例创建函数

创建了 ps::KVWorker 实例，KVStoreDist 之后会调用它的 Push， Pull， Wait 等函数。

```c++
  explicit KVStoreDist(bool use_device_comm)
      : KVStoreLocal(use_device_comm), ps_worker_(nullptr), server_(nullptr) {
    if (IsWorkerNode()) {
      int new_customer_id = GetNewCustomerId();
      ps_worker_ = new ps::KVWorker<char>(0, new_customer_id);
      ps::StartAsync(new_customer_id, "mxnet\0");
      if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(
          new_customer_id,
          ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
      }
    }
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
}
```

- 可以看出，<font color=red>如果当前是 **worker** node，KVStoreDist会创建一个 ps_worker_ 的成员，然后运行 ps::StartAsync , 再运行barrier `ps::Postoffice::Get()->Barrier( ,7)`</font>。  ps::StartAsync 与ps::Start 的唯一区别是，StartAsync中，当Postoffice和 van的 start初始化完成之后，不会执行Barrier(7)。因此上述创建函数中需要再手动设置barrier。

#### **InitImpl()**

这个函数被 Init() 调用。这个函数调用 **comm_->Init()**  和 **Push_**协助完成参数初始化，<font color=red>且规定Push_的 do_merge 为 false</font>

```c++
  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
    if (get_rank() == 0 && this->ps_worker_->get_customer()->customer_id() == 0) {
      Push_(keys, values, 0, false);
      // wait until the push is finished
      for (const int key : keys) {
        comm_buf_[key].WaitToWrite();
        compr_buf_[key].WaitToWrite();
      }
    } else {
      // do nothing
    }
    if (!ps::Postoffice::Get()->is_recovery()) {
      Barrier();
    }
  }
```



#### set_updater()

updater的设置是通过python端的函数定义来完成的，它通过ctype转换成为了c端的函数，并且通过pickle序列化为字符串传递给server。
当然，我们的主要注意力还是放在`push`和`pull`的实现上。

#### **PushImpl()**
<font color=red>调用 Push_ 并规定 do_merge 为 true</font>

```c++
   void PushImpl(const std::vector<int>& keys,
                 const std::vector<NDArray>& values,
                 int priority) override {
     Push_(keys, values, priority, true);
   }
```

 

#### **Push_()**

若 do_merge 为 true，push操作首先会通过`comm_-> Reduce()`操作。之后结果存储在`comm_buf_[key]`中。

调用 **EncodeDefaultKey** 函数将存储为`key : int`和`val : NDArray`形式的KVPair，转化为`PSKV`形式，该形式用于Push操作。<font color=red>同时也实现servers load balancing。</font>

之后会通过**PushDefault**方法完成操作，该方法定义了函数`push_to_servers`，将`comm_buf_[key]`作为输入，通过`Engine::Get()->PushAsync`方法完成**push**操作的异步执行（只是将任务发给Engine，由Engine完成调度）。Engine会在适当的时机执行`push_to_servers`. <font color=red>push_to_servers`函数调用了`ps_worker_`的`ZPush`方法来完成分布式的push操作。</font>

```c++
void Push_(const std::vector<int>& keys,
           const std::vector<NDArray>& values,
           int priority,
           bool do_merge) {
  // first aggregate the values over keys
  std::vector<int> uniq_keys;
  std::vector<std::vector<NDArray> > grouped_vals;
  GroupKVPairsPush(keys, values, &uniq_keys, &grouped_vals, false);

  for (size_t i = 0; i < uniq_keys.size(); ++i) {
    // merge over devices
    int key = uniq_keys[i];
    const auto& vals = grouped_vals[i];
    NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];

    const auto storage_type = merged.storage_type();
    auto &comm_buf = comm_buf_[key];
      
      
    if (merged.ctx().dev_mask() == cpu::kDevMask) {
      // Start of a push doesn't guarantee that the previous pushes are completed.
      // This shouldn't affect training of networks though because training involves
      // a sequence of push, pull, then push. This imposes ordering that the
      // second push happens after the first pull, and the pull happens after first push.
      comm_buf = merged;  // avoid memory copy
    } else {
      if (comm_buf.is_none()) {
        if (storage_type == kDefaultStorage) {
          comm_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
        } else {
          comm_buf = NDArray(storage_type, merged.shape(), pinned_ctx_, true, merged.dtype());
        }
      }
      CopyFromTo(merged, &comm_buf);
    }
      
      
    const int dtype = merged.dtype();
    const int num_bytes = mshadow::mshadow_sizeof(dtype);
    // push to servers
    if (storage_type == kDefaultStorage) {
      if (gradient_compression_->get_type() == CompressionType::kNone) {
        PSKV& pskv = EncodeDefaultKey(key, comm_buf.shape().Size(), num_bytes);
        PushDefault(key, comm_buf, pskv, priority);
      } 
    } 
  }
}
```

#### **EncodeDefaultKey()**

<font color=red>这里用 bigarray_bound_ 作为门槛执行tensor partition来实现 servers load balancing（参数分配， parameter assignment）， 调用ps::Postoffice::Get()->GetServerKeyRanges() 协助实现。</font>GetServerKeyRanges() 返回的是 ps::Postoffice 的成员 std::vector<Range> server_key_ranges\_，它的初始化如下,即把计算机的整数值域划分成 num\_servers_ 个互相独立的块。

```c++
Key kMaxKey = std::numeric_limits<Key>::max();    
for (int i = 0; i < num_servers_; ++i) {
      server_key_ranges_.push_back(Range(
          kMaxKey / num_servers_ * i,
          kMaxKey / num_servers_ * (i+1)));
}
```

函数把 key 转换成 ps_key, 使得每个ps_key 位于那个server的Range，这样之后的 ps::Van() 发送 tensor 时就可以根据 ps_key 的数值大小决定发给哪个server。

如果 num_arr_elems < bigarray\_bound\_，那么 pskv 保存这个tensor的 ps_key和长度信息。

如果 num_arr_elems > bigarray\_bound\_，那么把这个tensor切分成 num_servers 份，然后 pskv 保存切分后的小 tensors 们的 ps_key和长度信息。





```c++
 /**
   * \brief convert to pskv for parameter server
   * \param key
   * \param num_arr_elems number of elements in the value for key
   * \param num_bytes size of each element in number of bytes
   * \return PSKV used for both push and pull
   */
  inline PSKV& EncodeDefaultKey(const int key, const size_t num_arr_elems,
                                const int num_bytes) {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    size_t pskv_size = num_arr_elems * num_bytes;
    if (!pskv.keys.empty()) {
      CHECK_EQ(static_cast<size_t>(pskv.size), pskv_size)
        << "The value size cannot be changed " << pskv_size << ". Key is " << key;
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      const int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      // a simple heuristic for load balance
      if (num_arr_elems < bigarray_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        const int total_bytes = num_arr_elems * num_bytes;
        pskv.lens.push_back(total_bytes);
        pskv.size = total_bytes;
      } else {
        // parition it to all servers
        pskv.size = 0;
        for (int i = 0; i < num_servers; ++i) {
          size_t part_size =
            static_cast<size_t>(round(static_cast<double>(num_arr_elems)/num_servers*(i+1))) -
            static_cast<size_t>(round(static_cast<double>(num_arr_elems)/num_servers*i));
          ps::Key ps_key = krs[i].begin() + key;
          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          const int total_bytes = part_size * num_bytes;
          pskv.lens.push_back(total_bytes);
          pskv.size += total_bytes;
        }
      }
      CHECK_EQ(static_cast<size_t>(pskv.size), pskv_size);
    }
    return pskv;
  }
```

#### **PushDefault()**

这里ket-value pair 主体是 key和send_buf,  pskv 保存这个tensor是否被切割及切割后的信息。

传递给ps_worker_->ZPush()的参数是 pskv.keys, vals, pskv.lens

```c++

  void PushDefault(int key, const NDArray &send_buf, const PSKV& pskv, int priority) {
    auto push_to_servers =
        [this, key, pskv, send_buf](RunContext rctx, Engine::CallbackOnComplete cb) {
          const int dtype = send_buf.dtype();
          // convert to ps keys
          const size_t size = send_buf.shape().Size() * mshadow::mshadow_sizeof(dtype);
          char* data = static_cast<char *>(send_buf.data().dptr_);
          // do push. false means no delete
          ps::SArray<char> vals(data, size, false);
          int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype);
          CHECK_NOTNULL(ps_worker_)->ZPush(
              pskv.keys, vals, pskv.lens,
              cmd, [cb]() { cb(); });
        };
    Engine::Get()->PushAsync(
        push_to_servers,
        pinned_ctx_,
        {send_buf.var()},
        {},
        FnProperty::kNormal,
        priority,
        "KVStoreDistDefaultPush");
  }
```



#### **PullImpl()**

pull操作由该函数来完成，该函数会根据`keys`将**server**端的结果获取到对应的`NDArray`中。中间结果会保存在`comm_buf_[key]`中，这里由于之前`push`将该变量作为了输入，Engine在调度执行时会考虑到这点，保证所有对`comm_buf_[key]`的写操作(pull) 都在对它的读操作(push, 因为push将它作为了Engine的输入)完成之后。类似于`Push_`操作，Pull操作定义了函数`pull_from_servers`作为异步执行的函数，调用`PushAsync`发送给Engine。<font color=red>`pull_from_servers`函数调用了`ps_worker_`的`ZPull`方法来完成分布式的pull操作。</font>

```c++
  void PullImpl(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int priority, bool ignore_sparse) override {
    CHECK(ignore_sparse) << "dist kvstore pull doesn't support ignore_sparse=False";
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals, true);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = comm_buf_[key];
      const auto storage_type = grouped_vals[i][0]->storage_type();
      CHECK_EQ(storage_type, kDefaultStorage)
               << "Expected stype of value to be kDefaultStorage";
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_,
                           true, grouped_vals[i][0]->dtype());
      }
      auto pull_from_servers = [this, key, recv_buf](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        size_t size = recv_buf.shape().Size();
        const int dtype = recv_buf.dtype();
        const int num_bytes = mshadow::mshadow_sizeof(dtype);
        PSKV& pskv = (gradient_compression_->get_type() == CompressionType::kNone) ?
                      EncodeDefaultKey(key, size, num_bytes) :
                      EncodeCompressedKey(key, size, false, num_bytes);
        char* data = static_cast<char*> (recv_buf.data().dptr_);
        // false means not to delete data when SArray is deleted
        auto vals = new ps::SArray<char>(data, size * num_bytes, false);
        // issue pull
        RequestType mode = (gradient_compression_->get_type() != CompressionType::kNone) ?
                  RequestType::kCompressedPushPull : RequestType::kDefaultPushPull;
        const int cmd = GetCommandType(mode, dtype);
        CHECK_NOTNULL(ps_worker_)->ZPull(
          pskv.keys, vals, &pskv.lens, cmd, [vals, cb](){ delete vals; cb(); });
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {recv_buf.var()},
          FnProperty::kNormal,
          priority,
          "KVStoreDistDefaultStoragePull");

      comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
    }
  }
```



#### RunServer() 

```c++
void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    if (IsServerNode()) {
      server_ = new KVStoreDistServer();
      server_->set_controller(controller);
    }

    ps::StartAsync(0, "mxnet_server\0");
    if (!ps::Postoffice::Get()->is_recovery()) {
      ps::Postoffice::Get()->Barrier(0,
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }
    if (server_) server_->Run();
    ps::Finalize(0, true);
    delete server_;
    server_ = nullptr;
  }
```

这个函数会被初始 import mxnet 时启动 server和scheduler所用。如上所示，如果当前节点是server，那么会运行ps::StartAsync和Barrier用于初始化，用于server_->Run()用于起server主循环线程，最后ps::Finalize。而如果当前节点是scheduler，会运行ps::StartAsync和Barrier用于初始化, 然后就ps::Finalize。

### KVStoreDistServer （C++ class）

<img src="mxnet-role.jpg" alt="mxnet-role" style="zoom:20%;" />

- **server和scheduler的启动**：在我们通过python的`import mxnet`的时候，会有`import kvstore_server`，而导入该文件会运行`_init_kvstore_server_module()`:

  ```c++
  # python code    python/mxnet/kvstore_server.py
  def _init_kvstore_server_module():
      """Start server/scheduler."""
      is_worker = ctypes.c_int()
      check_call(_LIB.MXKVStoreIsWorkerNode(ctypes.byref(is_worker)))
      if is_worker.value == 0:
          kvstore = create('dist')
          server = KVStoreServer(kvstore)
          server.run()
          sys.exit()
  ```

  此函数会判断当前节点是否是**server 或 scheduler**节点，如果是就会创建python类KVStoreServer实例，并调用其`run()`函数，然后调用c++代码的`_LIB.MXKVStoreRunServer`，也就是c++类`KVStoreDist`的`RunServer`方法。**`RunServer`**方法中(代码如下)：

  - <font color=red>对于**server 和 scheduler**节点，都会运行 ps::StartAsync , 再运行barrier `ps::Postoffice::Get()->Barrier( ,7)`</font>
  - 如果当前节点是**server**，该方法会创建成员`server_ = new KVStoreDistServer();` ，并运行c++类 KVStoreDistServer的方法 Run(), 它是server的主循环线程， 具体细节见下面method  Run()。
  
  所以实际上这时的类KVStoreDist可以担任 scheduler和server两种角色
  
  ```c++
  # c++ code    src/kvstore/kvstore_dist.h
  # Class KVStoreDist
  void RunServer(const Controller& controller) override {
      CHECK(!IsWorkerNode());
      if (IsServerNode()) {
        server_ = new KVStoreDistServer();
        server_->set_controller(controller);
      }
  
      ps::StartAsync(0, "mxnet_server\0");
      if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(0,
          ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
      }
      if (server_) server_->Run();
      ps::Finalize(0, true);
      delete server_;
      server_ = nullptr;
    }
  ```
  
  

members:

```c++
  Executor exec_;
  ps::KVServer<char>* ps_server_;
```

methods:

#### constructor 创建函数

创建了ps::KVServer实例，并调用其 **set_request_handle()** 设置自己的两个函数为接收到  控制信息和数据信息的处理函数。这个KVStoreDistServer 也只调用了 ps::KVServer 的这一个函数

```c++
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<char>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandleEx, this, _1, _2, _3));
    sync_mode_ = false;
    gradient_compression_ = std::make_shared<GradientCompression>();
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
  }
```



#### Run() 

 仅仅只有一行`exec_.Start();`。这一行会调用`Executor exec_;`的`Start`方法。这个函数就是server的主循环线程。

这个Start函数会一直检查 exec\_的queue\_中是否有消息块，有的话就执行。而 ps-lite中独立信息接受处理线程Customer::Receiving() 会在接收到信息后通过层层调用函数handle，最终调用 KVStoreDistServer的 CommandHandle和 DataHandleEx 函数。而这两个会调用exec_.Exec() 函数，即将需要运行的函数封装在消息block中，并放在放在exec\_ 的 queue\_ 中。这样以来，原始主线程exec\_.Start()就可以执行它。

Executor的Start函数和Exec函数源码如下，当然也定义了Stop函数用于退出Start中的循环。

```c++
//c++ code            src/kvstore/kvstore_dist_server.h

class Executor {.......}  // executor runs a function using the thread called \ref Start()

//  start the executor
void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();}); // queue_为空，则等待被唤醒
      Block blk = std::move(queue_.front()); // 取出queue头元素
      queue_.pop();
      lk.unlock(); // 释放锁，给其他线程操作queue

      if (blk.f) { // 如果blk定义了一个function，则允许他
        blk.f();
        blk.p->set_value(); // 返回function的结果
      } else {
        blk.p->set_value(); break;
      }
      lk.lock(); // 获取锁，执行下一个循环
    }
```

调用`Executor`的`Exec`方法，会在`queue`中添加一个执行函数的`block`，代码如下:

```c++
 // let the thread called Start() to exec a function. threadsafe
void Exec(const Func& func) {
    Block blk(func); // 建立block
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one(); // 通知别的线程运行
    }
    fut.wait();
  }
```



#### **DataHandleDefault()**

该方法是默认的数据处理的方法，由于`DataHandleEx`被绑定为了数据的处理函数，当`RequestType`是`kDefaultPushPull`，就会调用该函数。它会根据传入的信息，提取对应的`key`，将对应的数据存储在`store_[key]`。如果从**worker**来的request类型是**push**，就会分两种情况运行。一种是初始化的时候，由于初始化同样通过调用**push**来完成，因此初始化的**push**只会将`store_[key]`设置为对应的值。另一种是初始化后，每一次的**push**都会进行相应的操作。这里每一次从任何一个**worker**来的某一个**key**的**push**操作，都会存储在`updates.merged`中，并且除了第一次的**push**，之后的**push**会进行`updates.merged += updates.temp_array;`也就是和之前的push相加。并且`ApplyUpdates`只会在**push**数达到**worker**的个数的时候，才会真正地进行。也只有在`ApplyUpdates`真正执行的时候才会将回复返回给**worker**。这样，就实现了同步。

```c++
  void DataHandleDefault(const DataHandleType type, const ps::KVMeta& req_meta,
                         const ps::KVPairs<char> &req_data,
                         ps::KVServer<char>* server) {
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }
    int key = DecodeKey(req_data.keys[0]);
    auto& stored = has_multi_precision_copy(type) ? store_realt_[key] : store_[key];
    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      size_t ds[] = {(size_t) req_data.lens[0] / mshadow::mshadow_sizeof(type.dtype)};
      mxnet::TShape dshape(ds, ds + 1);
      TBlob recv_blob;
      MSHADOW_REAL_TYPE_SWITCH(type.dtype, DType, {
        recv_blob = TBlob(reinterpret_cast<DType*>(req_data.vals.data()), dshape, cpu::kDevMask);
      })
      NDArray recved = NDArray(recv_blob, 0);
      if (stored.is_none()) {
        // initialization
        stored = NDArray(dshape, Context(), false,
                         has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
        CopyFromTo(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else {
        auto &updates = update_buf_[key];
        if (sync_mode_ && updates.merged.is_none()) {
          updates.merged = NDArray(dshape, Context(), false,
                                   has_multi_precision_copy(type) ? mshadow::kFloat32 : type.dtype);
        }
        if (updates.request.empty()) {
          if (sync_mode_) {
            CopyFromTo(recved, updates.merged);
          } else {
              updates.temp_array = recved;
            }
          }
        } else {
          CHECK(sync_mode_);
            updates.merged += recved;
          }
        }
        updates.request.push_back(req_meta);
        ApplyUpdates(type, key, &updates, server);
      }
    } else {
      DefaultStorageResponse(type, key, req_meta, req_data, server);
    }
  } 
```



### CommCPU

继承自 Comm

用到的环境变量：

- MXNET_KVSTORE_REDUCTION_NTHREADS

  Values: Int （default=4）

  The number of CPU threads used for summing big arrays.

- MXNET_KVSTORE_BIGARRAY_BOUND

  Values: Int (default=1000000)

  The minimum size of a “big array”. When the array size is bigger than this threshold, MXNET_KVSTORE_REDUCTION_NTHREADS threads are used for reduction.

  This parameter is also used as a load balancer in kvstore. It  controls when to partition a single weight to all the servers. If the  size of a single weight is less than MXNET_KVSTORE_BIGARRAY_BOUND then,  it is sent to a single randomly picked server otherwise it is  partitioned to all the servers. 这里即**bigarray_bound_**

members：

```c++
nthread_reduction_ = dmlc::GetEnv("MXNET_KVSTORE_REDUCTION_NTHREADS", 4);
bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
```



methods：

- Init()

  ```c++
    void Init(int key, const NDArrayStorageType stype, const mxnet::TShape& shape,
              int type = mshadow::kFloat32) override {
      // Delayed allocation - the dense merged buffer might not be used at all if push()
      // only sees sparse arrays
      bool delay_alloc = true;
      merge_buf_[key].merged = NDArray(shape, pinned_ctx_, delay_alloc, type);
    }
  ```

- Reduce()

  调用 ReduceSumCPU()

  ```c++
    const NDArray& Reduce(int key, const std::vector<NDArray>& src,
                          int priority) 
  ```

- ReduceSumCPU()

  调用 ReduceSumCPUImpl()

  ```c++
   private:
    // reduce sum into val[0]
    inline void ReduceSumCPU(const std::vector<NDArray> &in_data) {
      MSHADOW_TYPE_SWITCH(in_data[0].dtype(), DType, {
        std::vector<DType*> dptr(in_data.size());
        for (size_t i = 0; i < in_data.size(); ++i) {
          TBlob data = in_data[i].data();
          CHECK(data.CheckContiguous());
          dptr[i] = data.FlatTo2D<cpu, DType>().dptr_;
        }
        size_t total = in_data[0].shape().Size();
        ReduceSumCPUImpl(dptr, total);
      });
    }
  ```

  

- ReduceSumCpuImpl()

  调用 ReduceSumCPU(dptr, offset, size)

  这里用 **bigarray_bound_** 完成对大tensor的切分，从而用多线程去 reduce

  ```c++
    template<typename DType>
    inline void ReduceSumCPUImpl(std::vector<DType*> dptr, size_t total) {
      const size_t step = std::min(bigarray_bound_, static_cast<size_t>(4 << 10));
      long ntask = (total + step - 1) / step; // NOLINT(*)
      if (total < bigarray_bound_ || nthread_reduction_ <= 1) {
        ReduceSumCPU(dptr, 0, total);
      } else {
        #pragma omp parallel for schedule(static) num_threads(nthread_reduction_)
        for (long j = 0; j < ntask; ++j) { // NOLINT(*)
          size_t k = static_cast<size_t>(j);
          size_t begin = std::min(k * step, total);
          size_t end = std::min((k + 1) * step, total);
          if (j == ntask - 1) CHECK_EQ(end, total);
          ReduceSumCPU(dptr, begin, static_cast<index_t>(end - begin));
        }
      }
    }
  ```

- ReduceSumCPU(dptr, offset, size)

  ```c++
    template<typename DType>
    inline static void ReduceSumCPU(
        const std::vector<DType*> &dptr, size_t offset, index_t size)
  ```

  


## 4 gluon: "imperative" training 

### image_classification.py 调用gluon

- 调用形式

```python
    from mxnet.gluon.model_zoo import vision as models
    net = models.get_model(model, **kwargs)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            optimizer_params={'learning_rate': opt.lr,
                                              'wd': opt.wd,
                                              'momentum': opt.momentum,
                                              'multi_precision': True},
                            kvstore=kv)
```

net 是一个 网络实例，例如 VGG 实例， 类 VGG继承自 HybridBlock

- 重要参数

  storage_type of NDArray: default

  compression: no



classes:

### Trainer (python class)

file: python/mxnet/gluon/trainer.py

members:

 **self.\_params** : list of Parameter instances

methods:

- init()

  params:  list of Parameter instances

  这里用  self._param2idx 保存 Parameter 的 name 与 int 数字标签的映射关系。int 数字作为后面 kvstore 中key-value pair 的 key，而 value 则是对应的 Parameter.list_grad()。

  ```python
  def __init__(self, params, optimizer, optimizer_params=None, kvstore='device',
               compression_params=None, update_on_kvstore=None):
        ......
        if isinstance(params, (dict, ParameterDict)):
        params = list(params.values())
        self._params = []
        # parameters to initialize on the kvstore
        self._param2idx = {}
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    "got list of %s."%(type(param)))
            self._param2idx[param.name] = i
            self._params.append(param)
            param._set_trainer(self)
  ```

  

- step() 

  训练一个iteration，包括通信（\_allreduce_grads）和计算（\_update）

  ```python
          rescale_grad = self._scale / batch_size
          self._check_and_rescale_grad(rescale_grad)
  
          if not self._kv_initialized:
              self._init_kvstore()
          if self._params_to_init:
              self._init_params()
  
          self._allreduce_grads()
          self._update(ignore_stale_grad)
  ```


- \_init_kvstore()

  use  `kvstore, update_on_kvstore = _create_kvstore(......)`

- \_init_params()

  注意，这里 调用kvstore init() pull() 传输的是 Parameter.\_data; 而之后训练时传输的是 Parameter.\_grad

  self.\_params_to_init 与 self.\_params 相同，都是 Parameter 的 list 。

  ```python
          params_to_init = []
          if self._kvstore:
              for param in self._params_to_init:
                  if param._deferred_init:
                      params_to_init.append(param)
                  else:
                      param_arrays = param._check_and_get(param._data, list)
                      idx = self._param2idx[param.name]
                      self._kvstore.init(idx, param_arrays[0])
                      if param._stype == 'default':
                          self._kvstore.pull(idx, param_arrays, priority=-idx)
  ```

  

- \_allreduce_grads()

  这里之后的push pull操作都会传递给Engine。对于相同key的两个操作，Engine会先执行完成push，才会执行pull。而之后的 c++类 KVStoreDist 和KVStoreDistServer会确保server聚合完了所有worker关于这个key的梯度后才会判定push执行完成。之后Engine再执行pull即可获得正确的同步梯度。

  ```python
          if self._kvstore:
              for i, param in enumerate(self._params):
                  if param.grad_req != 'null':
  
                      self._kvstore.push(i, param.list_grad(), priority=-i)
                      if not self._update_on_kvstore:
                          self._kvstore.pull(i, param.list_grad(), priority=-i,
                                             ignore_sparse=self._distributed)
  ```

- \_update()

  ```python
          updates = [[] for _ in self._updaters]
  
          for i, param in enumerate(self._params):
              if param.grad_req == 'null':
                  continue
              ......
  
              for upd, arr, grad in zip(updates, param.list_data(), param.list_grad()):
                  if not ignore_stale_grad or arr._fresh_grad:
                      upd.append((i, grad, arr))
                      arr._fresh_grad = False
  
          if not (self._kvstore and self._update_on_kvstore):
              for updater, upd in zip(self._updaters, updates):
                  if upd:
                      i, w, g = zip(*upd)
                      updater(i, w, g)
  ```

## 5 update_on_kvstore

- 由环境变量 MXNET_UPDATE_ON_KVSTORE  控制, 用来决定是否在ps端更新参数，这个参数在python的KVStore类初始化时被使用。如果该变量为True，那么模型参数的更新发生在ps端；否则，模型参数的更新发生在worker端，ps端只做梯度的聚合操作（这种情况下，paramerter server是不是就变成了gradient server？）。

- 只有在同步训练模式下，我们才能设置`update_on_kvstore=False`，异步训练并不支持在worker端更新参数。

- 在`update_kv_store=True`的情况下，我们需要告诉ps端训练过程中使用的优化器是什么，因此要调用`kvstore.set_optimizer`把优化器从worker端传给ps端。**<font color=red>具体为 python的KVStore类用pickle。dumps把optimizer对象序列化，其c++类通过ps-lite的SimpleApp.Request函数把序列化后的字符串发送给PS端。PS端的python的KVStoreServer类用pickle.loads将其反序列化得到优化器对象</font>**。

同步训练模式下调用Trainer类的参数同步有两种方式，如下图（最左一列为Trainer类内部的与参数同步有关的函数，右边格子中为这个函数具体执行了哪些操作。这些函数只工作在worker端。当update_on_kvstore=true时，updater操作在PS端完成，因此worker端没有updater操作，pull回来的直接就是全局参数）：

- 图上部的一种模式为直接调用Trainer.step()； 
- 图下部的一种模式为worker端需要对聚合后的梯度进行一定操作（如图中的clip操作），因此只能用update_on_kvstore=false 模式 ， 需要手动调用Trainer的两个同步函数。

<img src="update_on_kvstore.jpg" alt="update_on_kvstore" style="zoom:20%;" />


## 6 tensor structure and transmission



### Mxnet role main classes

<img src="mxnet-role.jpg" alt="mxnet-role" style="zoom:20%;" />

### Parameter (python class)

file: python/mxnet/gluon/parameter.py

members:

- _data : list, 只包含一个 NDArray 元素
- _grad : list, 只包含一个 NDArray 元素

methods:

- list_grad()

  ```python
      def list_grad(self):
          """Returns gradient buffers on all contexts, in the same order
          as :py:meth:`values`."""
          if self._data is not None and self._grad is None:
              raise RuntimeError(
                  "Cannot get gradient array for Parameter '%s' " \
                  "because grad_req='null'"%(self.name))
          return self._check_and_get(self._grad, list)
      
  ```



### HybridBlock(python class)

继承自 Block

file: python/mxnet/gluon/block.py

methods:

- collect_params()

  ```python
    def collect_params(self, select=None):
        """Returns a :py:class:`ParameterDict` containing this :py:class:`Block` and all of its
        children's Parameters(default), also can returns the select :py:class:`ParameterDict`
        which match some given regular expressions.

        For example, collect the specified parameters in ['conv1_weight', 'conv1_bias', 'fc_weight',
        'fc_bias']::

            model.collect_params('conv1_weight|conv1_bias|fc_weight|fc_bias')

        or collect all parameters whose names end with 'weight' or 'bias', this can be done
        using regular expressions::

            model.collect_params('.*weight|.*bias')

        Parameters
        ----------
        select : str
            regular expressions

        Returns
        -------
        The selected :py:class:`ParameterDict`
        """
        # We need to check here because blocks inside containers are not supported.
        self._check_container_with_block()
        ret = ParameterDict(self._params.prefix)
        if not select:
            ret.update(self.params)
        else:
            pattern = re.compile(select)
            ret.update({name:value for name, value in self.params.items() if pattern.match(name)})
        for cld in self._children.values():
            ret.update(cld.collect_params(select=select))
        return ret
  ```



### key-value pair

- python层面
Trainer.\_params:  list of Parameter instances. Trainer._param2idx 保存 Parameter 的 name 与 int 数字标签的映射关系。
**key** : int 数字.
**value** :  Parameter.list_grad():  是一个list, 只包含一个 NDArray 元素

- c++层面

  以 KVStore 的init和push输入参数为例：

  **std::vector<int>** v_keys(num);
  **std::vector<NDArray>**  v_vals(num);

### tensor size (Parameter instance)

下图中左边理论分析中vgg16最大的那个tensor（102M）在有图中mxnet 中只有 4096*512，不知道为什么？

![vgg16-tensor](vgg16-tensor.png)



**原因： 数据集cifar10 的单个输入图片长宽小于标准数据集（imagenet）的图片输入。数据集会影响模型大小（单个图片的长宽对应于模型中的某些节点数量）。batch-size不会影响模型大小。**

用数据集 caltech101 就是138M数据量了，这时 batch size 只能为32，否则out of memory



### All workers'  init behaviours       

- First worker(rank:0, id: 9) will do Init(Push) & Pull at first. 

  (When server receives the first pushed gradients, it stores them as merged gradients)

- Other workers will do Init(do nothing) & Pull at first.
  

### Worker Init function path

**main path** :

**<font color=green>KVStore.init() (python)  --> KVStoreDist.InitImpl()  -->  KVStoreDist.Push_() --> KVStoreDist.PushDefault()</font>--> ps:: KVWorker.ZPush() -->  ps:: KVWorker.Send() --> <font color=red>Postoffice::Get()->van()->Send(msg) --> ZMQVan.SendMsg()</font>**

- 其中上述path中绿色函数中的 tensor存在形式为 {int key, NDArray value} ,绿色和黑色函数中处理一个完整tensor，红色函数则只处理一个切分后的小tensor。
- **<font color=green> KVStoreDist.InitImpl()</font>**中会判断，只有当`get_rank() == 0 && this->ps_worker_->get_customer()->customer_id() == 0)` ，即当前worker是 rank=0，id=9 的第一个worker时，才会真正执行**Push_** , 否则 do nothing。即所有的worker的 init 只有第一个worker真正的把初始数据传输给了server。

### Worker Push function path 

here , a tensor =  a key-value pair = a parameter group of one NN layer

**main path** :

**<font color=green>KVStore.push() (python)  --> KVStoreDist.PushImpl()  -->  KVStoreDist.Push_() --> KVStoreDist.PushDefault()</font>--> ps:: KVWorker.ZPush() -->  ps:: KVWorker.Send() --> <font color=red>Postoffice::Get()->van()->Send(msg) --> ZMQVan.SendMsg()</font>**



- 其中上述path中绿色函数中的 tensor存在形式为 {int key, NDArray value} ,绿色和黑色函数中处理一个完整tensor，红色函数则只处理一个切分后的小tensor（这个切分是针对多个PSes的load balancing）。

- **<font color=green>KVStoreDist.Push_()</font>**函数中借助EncodeDefaultKey() 函数分割tensor并保存分割信息到 pskv 中。 此时 pskv.keys 中包含一个或 num_server 个 ps_key

-  **<font color=green>PushDefault()</font>** 创建push_to_servers传递给Engine异步执行。push_to_servers调用 **ZPush()** 时传入的参数为 ZPush(pskv.keys, **vals**, pskv.lens, cmd, [cb]() { cb(); })。<font color=red>这里创建的 cb 会传递给 ps::KVWorker, 等待 worker收到server回复说明server已经聚合完成了所有worker关于这个key的梯度后，worker才会执行cd，即告诉Engine 当前push_to_servers 已经执行完成</font>。 **vals & cmd** 是这样生成的：
  
  ```c++
  int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype)        
  // send_buf 即为 NDArray value
  const int dtype = send_buf.dtype();
  // convert to ps keys
  const size_t size = send_buf.shape().Size() * mshadow::mshadow_sizeof(dtype);
  char* data = static_cast<char *>(send_buf.data().dptr_);
  // do push. false means no delete
  ps::SArray<char> vals(data, size, false);
  ```
  
- **ps::KVWorker.ZPush()** <font color=red>根据timestamp保存cb</font>，并用如下数据结构继续调用 **Send(ts, true, cmd, kvs)**。其中第二个参数表示当前操作为push，这个bool信息会在 Send() 中被写入 msg.meta.push 中。这时 kvs.keys 中包含一个或 num_server 个 ps_key

    ```c++
    KVPairs<Val> kvs;
    kvs.keys = keys;
    kvs.vals = vals;
    kvs.lens = lens;
    ```

- **ps::KVWorker.Send()** 调用DefaultSlicer根据kvs.keys 的 ps_key 的数值大小进行切分，得到包含切分后的小tensor的 <bool, KVPairs>的 vector, 保存在 sliced 中。<font color=red>原来的 kvs.keys 中包含一个或 num_server 个 ps_key，但是 sliced 中一定有且仅有有 num_servers 个 <bool , KVPairs></font>，每一个在 vector中的位置对应要发送给的 server rank。bool 表示要不要发。这时以及经过切分，所以每个 KVPairs.keys 只包含一个 ps_key。

  ```c++
  using SlicedKVs = std::vector<std::pair<bool, KVPairs<Val>>>;  
  SlicedKVs sliced;
  slicer_(kvs, Postoffice::Get()->GetServerKeyRanges(), &sliced);
  ```

  之后 **ps::KVWorker.Send()**  把 sliced 封装成数个 msg（每个msg 包含 serverID），调用 **<font color=red>Postoffice::Get()->van()->Send(msg)</font>** 发送给对应的server。 封装 msg 过程如下：

  ```c++
    for (size_t i = 0; i < sliced.size(); ++i) {
      const auto& s = sliced[i];
      if (!s.first) continue;
      Message msg;
      msg.meta.app_id = obj_->app_id();
      msg.meta.customer_id = obj_->customer_id();
      msg.meta.request     = true;
      msg.meta.push        = push;
      msg.meta.head        = cmd;
      msg.meta.timestamp   = timestamp;
      msg.meta.recver      = Postoffice::Get()->ServerRankToID(i);
      const auto& kvs = s.second;
      if (kvs.keys.size()) {
        msg.AddData(kvs.keys);
        msg.AddData(kvs.vals);
        if (kvs.lens.size()) {
          msg.AddData(kvs.lens);
        }
      }
      Postoffice::Get()->van()->Send(msg);
    }
  ```


### Worker Pull function path

main path:

**<font color=green>KVStore.pull() (python)  --> KVStoreDist.PullImpl()  </font>--> ps:: KVWorker.ZPull() ->Pull_() --> ps:: KVWorker.Send() --> <font color=red>Postoffice::Get()->van()->Send(msg) --> ZMQVan.SendMsg()</font>**

- 其中上述path中绿色函数中的 tensor存在形式为 {int key, NDArray value} ,绿色和黑色函数中处理一个完整tensor，红色函数则只处理一个切分后的小tensor。

- **<font color=green>KVStoreDist.PullImpl()</font>** 函数创建pull_from_server传递给Engine执行。pull_from_servers中借助EncodeDefaultKey() 函数分割tensor并保存分割信息到 pskv 中。 此时 pskv.keys 中包含一个或 num_server 个 ps_key。调用 **ZPull()** 所传入的 cmd 如下：

  ```c++
  int cmd = GetCommandType(RequestType::kDefaultPushPull, dtype) 
  ```

  调用ZPull() 也会传入cd，用于之后的ps::KVWorker执行。

- **<font color=green>KVStoreDist.PullImpl()</font>** 调用 ps:: KVWorker.ZPull() -> Pull\_() 的输入数据结构和之前的 ZPush 一样 ，**ps:: KVWorker.Pull_()** 定义了 callback。 

- **ps:: KVWorker.Pull_(key，value)** <font color=red>定义了 cb_0, 并根据timestamp保存起来</font>。再调用 ***Send(ts, false, cmd, kvs)** ; 其中第二个参数表示当前操作为pull，这个 bool 信息会在 Send() 中被写入 msg.meta.push 中。之后worker接受到server回复的pull梯度数据后执行cb_0。cb_0 中从 recv_kvs_[ts] 取出接收到的 server数据，赋给value，并且执行之前传递下来的cd，即通知mxnet Engine 这个pull已成功完成。

### Worker receiving function path

对于 worker 的init(key, value) 和 push(key, value) 传输，server会回复原来的请求信息，即 server->Response(req_meta)。这个回复不包括数据，仅告知 worker 传输已经成功。并且 对于 push 操作来说，server 只有接收到所有worker关于这个key的数据后才会Response。这意味着，当 worker收到 关于 push(key, value) 的response时，server 端已经聚合了所有worker关于这个key的梯度数据。

对于 worker 的pull(key, value) 传输，server会直接回复聚合好了的梯度：server->Response(req_meta, kvpairs)

所以worker 的receiving应该处理这两种不同的回复。

main path:

- **ZMQVan.RecvMsg() --> Van.Receiving() --> Customer.Accept() --> Customer.recv_queue_.Push(msg)**

- **Customer.Receiving() --> ps::KVWorker.Process()**

  KVWorker.Process() 是处理 server Response 信息的函数，部分代码如下。之前worker执行init，push，pull操作时，都会在发送时根据timestamp保存cd或 cd_0。

  - 对于之前的init 和 push操作，Process 收到对应server的Response后（以下代码看起来是对比timestamp时操作接收到的Response数 和 server数，实际上并不需要等待所有的server回复，因为ps::KVWorker.Send()以及把没有发送请求过去的server剔除了 ），会运行 cb。运行 cb 即告诉mxnet Engine 之前的 Push 操作完成。
  - 对于之前的pull操作，Process 收到对应server的Response后，包数据保存在recv_kvs_[ts]中，并执行 cd_0。cb_0 内部再执行 cd，通知Engine。

  ```c++
  template <typename Val>
  void KVWorker<Val>::Process(const Message& msg) {
    if (msg.meta.simple_app) {
      SimpleApp::Process(msg); return;
    }
    // store the data for pulling
    int ts = msg.meta.timestamp;
    if (!msg.meta.push && msg.data.size()) {
      CHECK_GE(msg.data.size(), (size_t)2);
      KVPairs<Val> kvs;
      kvs.keys = msg.data[0];
      kvs.vals = msg.data[1];
      if (msg.data.size() > (size_t)2) {
        kvs.lens = msg.data[2];
      }
      mu_.lock();
      recv_kvs_[ts].push_back(kvs);
      mu_.unlock();
    }
  
    // finished, run callbacks
    if (obj_->NumResponse(ts) == Postoffice::Get()->num_servers() - 1)  {
      RunCallback(ts);
    }
  }
  ```

   

### Server receiving function path

main path:

- **ZMQVan.RecvMsg() --> Van.Receiving() --> Customer.Accept() --> Customer.recv_queue_.Push(msg)**

- **Customer.Receiving() --> ps::KVServer.Process() --> KVStoreDistServer.DataHandleEx() -->  KVStoreDistServer.DataHandleDefault()**

KVStoreDistServer.DataHandleDefault()中：

- 如果 req_meta.push 为 true（push 操作）：聚合，如果是worker  init ，则直接Response(req_meta)。如果是一般的worker push操作，ApplyUpdates()会在聚合次数等于 num_servers 时回复Response(req_meta给worker。
- 如果 req_meta.push 为 false（pull 操作）：执行 DefaultStorageResponse()直接返回聚合好了的梯度数据。

### Worker command to server function path

worker send path：

- KVStoreDDist.SendCommandToServer()  --> ps::SimpleApp.Request() --> ZMQVan.Send()

server receiving path:

- ZMQVan.RecvMsg() --> Van.Receiving() --> Customer.Accept() --> Customer.recv_queue_.Push(msg)
- Customer.Receiving() --> ps::KVServer.Process() --> ps::SimpleApp.Process() --> KVStoreDistServer.CommandHandle()

## 7 Copy in Push and Pull

1) Push时， KVStoreDist::Push_()  中先把要push的数据copy到 comm_buf\_[key] 中。

2） Pull时， KVStoreDist::PullImpl() 调用 ps::KVWorker::ZPull() 把pull回来的数据 copy到 comm_buf\_[key] , 再调用 comm\_.Broadcast() 把comm_buf\_[key] 的数据 copy 到 真正的 vals中。

3） Push和Pull再调用 Engine::PushAsync() 时设定的依赖变量是 comm_buf\_[key]  ， 所以可以告诉engine正确的依赖关系。