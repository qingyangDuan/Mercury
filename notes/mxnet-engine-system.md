**Mxnet Engine Async process task（c++）**

files location：include/mxnet/engine.h    src/engine/*






## 1 Dependency Engine Overview
You can use MXNet’s engine not only for deep learning, but for any domain-specific problem. It’s designed to solve a general problem: execute a bunch of functions following their dependencies. Execution of any two functions with dependencies should be serialized. To boost performance, functions with no dependencies *can* be executed in parallel. For a general discussion of this topic, see the [Note on Dependency Engine](https://mxnet-bing.readthedocs.io/en/latest/architecture/note_engine.html).

other useful links:
http://www.liuhaihua.cn/archives/348939.html
https://blog.csdn.net/chaojichaoachao/article/details/51997174?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-7.baidujs&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-7.baidujs

### Main Interface

The following API is the core interface for the execution engine:

```
    virtual void PushSync(Fn exec_fun, Context exec_ctx,
                          std::vector<VarHandle> const& const_vars,
                          std::vector<VarHandle> const& mutate_vars) = 0;
```

This API allows you to push a function (`exec_fun`), along with its context information and dependencies, to the engine. `exec_ctx` is the context information in which the `exec_fun` should be executed. `const_vars` denotes the variables that the function reads from, and `mutate_vars` are the variables to be modified. Regardless of the details that we’ll explain later, the engine guarantees following order:

**The execution of any two functions when one of them modifies at least one common variable is serialized in their push order.**  (For details, see "VarHandle" part)

- 可见，输入的 **const_vars** 和 **mutate_vars** 并不是 vector\<NDArray>（tensor 就是 NDArray实例），而是 vector\<**NDArray.var()**>。显然 NDArray.var() 返回的就是 VarHandle。

在分布式训练的一次迭代中，各个func用Pushsync放进Engine中。Engine根据 const_vars 和 mutate_vars 来确定各个func的依赖关系，并对无依赖的func执行一定的并行化以加速执行。各个func执行时互相依赖关系的例子：

<img src="pics/engine-example.jpg" style="zoom:20%;" />


### Function

In the `AsyncFn` function, you can pass the heavy part to your own threads and safely exit the body of the function. The engine doesn’t consider the function finished until the `Callback` function is called.

```c++
    struct RunContext {
        // stream pointer which could be safely cast to
        // cudaStream_t* type
        void *stream;
    };
    using Callback = std::function<void()>;
    using AsyncFn = std::function<void(RunContext, Callback)>;
```

In the `AsyncFn` function, you can pass the heavy part to your own threads and safely exit the body of the function.<font color=red> **The engine doesn’t consider the function finished until the `Callback` function is called.**</font>

### Context

You can specify the `Context` of the function to be executed within. This usually includes whether the function should be run on a CPU or a GPU, and if you specify a GPU, which GPU to use. `Context` is different from `RunContext`. `Context` contains device type (gpu/cpu) and device id, while `RunContext` contains information that can be decided only during runtime, for example, on which stream the function should be executed.

### VarHandle

`VarHandle` is used to specify the dependencies of functions. The MXNet engine is designed to be decoupled from other MXNet modules. So `VarHandle` is like an engine-provided token you use to represent the external resources the functions can use or modify. It’s designed to be lightweight, so creating, deleting, or copying a variable incurs little overhead. Upon pushing the functions, you need to specify the variables that will be used (immutable) in the `const_vars` vector, and the variables that will be modified (mutable) in the `mutate_vars` vector. 

<font color=red>**The engine uses one rule for resolving the dependencies among functions:** </font>

> <font color=red>**The execution of any two functions when one of them modifies(mutate) at least one common variable (const_vars + mutate_vars) is serialized in their push order.** </font>

For example:

- if `Fn1` and `Fn2` both mutate `V2`,  then `Fn1` and`Fn2` is guaranteed to be executed in their push order.
- if `Fn1`  mutate `V2` and `Fn2` use `V2`,  then `Fn1` and`Fn2` is guaranteed to be executed in their push order.
- if `Fn1` and `Fn2` both use `V2`, their actual execution order could be random. 

This design allows the engine to schedule *state-mutating* operations. For example, the weight update function in DNN can now use the `+=` operator to update the weights in place, rather than generating a new weight array each time.

To create a variable, use the `NewVar()` API. To delete a variable, use the `PushDelete` API.

### Push and Wait

*All `Push` APIs are asynchronous.* The API call returns immediately regardless of whether the pushed `Fn` is finished. This allows the engine to start computing at the same time the user thread is pushing functions. All `Push` APIs are not thread-safe. To be specific, only one thread should make engine API calls at a time.

If you want to wait for a specific `Fn` to be finished, include a callback function in the closure, and call the function at the end of your `Fn`.

If you want to wait for all `Fn`s that involve (use or mutate) a certain variable to finish, use the `WaitForVar(var)` API.

If you want to wait for all pushed `Fn`s to finish, use the `WaitForAll()` API.

### Save Object Creation Cost

In some cases, you need to push several functions to the engine for a long period of time. If the computation of these functions is light, the overhead of copying lambdas and creating use/mutate variable lists becomes relatively high. We provide an API to create an `OprHandle` beforehand:

```
    virtual OprHandle NewOperator(AsyncFn fn,
                                  std::vector<VarHandle> const& const_vars,
                                  std::vector<VarHandle> const& mutate_vars) = 0;
```

You can keep pushing the `OprHandle` without repeatedly creating them:

```
    virtual void Push(OprHandle op, Context exec_ctx) = 0;
```

To delete it, call the `DeleteOperator(OprHandle op)` API. Ensure that the operator has finished computing before calling this API.

## 2 classes

### Engine (abstract)

- Engine::Get()

  根据Engine类型返回 NaiveEngine,  ThreadedEnginePooled,  或 ThreadedEnginePerDevice(default choice) 的实例指针

### NativeEngine (继承自Engine)

### ThreadedEngine (abstract) (继承自Engine)

methods:

- PushAsync()

  调用 Push()
  
  ```c++
    /*!
     * \brief Push an asynchronous operation to the engine.
     * \param exec_fun Execution function, this function takes a parameter
     *                 on_complete that must be called when the execution
     *                 completes.
     * \param exec_ctx Execution context.
     * \param const_vars The variables that current operation will use but not
     *                   mutate.
     * \param mutable_vars The variables that current operation will mutate.
     * \param prop Property of the function.
     * \param priority Priority of the action, as hint to the engine.
     * \param opr_name The operator name.
     * \param wait Whether this is a WaitForVar operation
     */
  void ThreadedEngine::PushAsync(AsyncFn fn, Context exec_ctx,
                                 std::vector<VarHandle> const& const_vars,
                                 std::vector<VarHandle> const& mutable_vars,
                                 FnProperty prop,
                                 int priority,
                                 const char* opr_name,
                                 bool wait) {
    ......
    ThreadedOpr *opr = NewOperator(std::move(fn), const_vars, mutable_vars, prop, opr_name, wait);
    opr->temporary = true;
    const bool profiling = profiler_->IsProfiling(profiler::Profiler::kImperative);
  Push(opr, exec_ctx, priority, profiling);
  }
  ```
  
  

- Push()

  ```c++
  void ThreadedEngine::Push(OprHandle op, Context exec_ctx, int priority, bool profiling) {
    BulkFlush();
  
    ThreadedOpr* threaded_opr = ThreadedOpr::CastFromBase(op);
    OprBlock* opr_block = OprBlock::New();
    opr_block->opr = threaded_opr;
  
    opr_block->wait.store(static_cast<int>(
        threaded_opr->const_vars.size() +
        threaded_opr->mutable_vars.size() + 1));
    opr_block->ctx = exec_ctx;
    opr_block->priority = priority;
    opr_block->profiling = profiling;
    ++pending_;
    // Add read dependencies.
    for (auto&& i : threaded_opr->const_vars) {
      i->AppendReadDependency(opr_block);
    }
    // Add write dependencies.
    for (auto&& i : threaded_opr->mutable_vars) {
      i->AppendWriteDependency(opr_block);
    }
    if (opr_block->decr_wait() == 0) {
      this->PushToExecute(opr_block, true);
    }
  }
  ```

  

  

### ThreadedEnginePerDevice (继承自ThreadedEngine)

methods:

- PushToExecute()

  

### ThreadedEnginePooled (继承自ThreadedEngine)





## 3 system

refer to https://mxnet.apache.org/versions/1.8.0/api/architecture/overview.html#operators-in-mxnet

http://shuokay.com/2017/10/04/mxnet-add-op-in-backend/

https://mxnet.apache.org/versions/1.8.0/api/faq/add_op_in_backend



- Comm OP 

  - 具体如何执行由 ps-lite实现。

  - operator 由 c++ class KVStoreDist（src/kvstore/*.h  & *.cc）产生并push给Engine, 由Engine调度执行。

- Comp OP

  - 所有的 comp operator （e.g. FullyConnect, Conv ) 具体如何执行都由c++实现（src/operator/*），

  - 使用 python API时，在 `import MXNet as mx` 的时候，在后端每个operator都会链接上对应的两个 Python function，  分别是用于 imperative programming 的 `MXNet.ndarray.opname` 和用于 symbolic programming 的 `MXNet.symbol.opname`。
  - operator 由 mxnet::Executor（src/executor/*）push给Engine， 由Engine调度执行。



## 4 symbolic && imperative

https://mxnet.apache.org/versions/1.1.0/tutorials/index.html

MXNet has two primary high-level interfaces for its deep learning engine: **the Gluon API** and **the Module API**.The difference between the two is an imperative versus symbolic programming style. 

- Gluon makes it easy to prototype, build, and train deep learning models without sacrificing training speed by enabling both (1) intuitive imperative Python code development and (2) faster execution by automatically generating a symbolic execution graph using the hybridization feature.



#### 三种模式

https://mxnet.incubator.apache.org/versions/1.8.0/api/python/docs/tutorials/packages/gluon/blocks/hybridize.html

- symbolic : 主要用 model.fit() 训练
- imperative ： 用 gluon.Trainer 训练
- hybrid： 用 gluon.Trainer 训练， 且执行 net.hybridize()