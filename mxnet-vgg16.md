

# VGG16 in mxnet

## Model Size

- 数据集cifar10时： 总大小38M左右，如下图右边部分，主要是由于原来理论上一个102M的tensor变成了2M左右。

![vgg16-tensor](vgg16-tensor.png)

- 数据集为caltech101 时， 总大小达到上图左边的理论大小，138M左右。不过用这个数据集，在1080ti GPU（11G显存）上跑，只能跑 batch-size为32的训练。batch-size再大就是报错显存不足。





## Forward function roadmap

**net() -> HybridBlock.forward() -> VGG.hybrid_forward()**

net is a instance of class VGG

### image_classification.py 中的 **net()** :

```python
net=VGG()  #class VGG(HybridBlock)
for i, batch in enumerate(train_data):
  data =...
  label =...
  Ls = []
  with ag.record():
      for x, y in zip(data, label):
          z = net(x)
          L = loss(z, y)
          Ls.append(L)
          ag.backward(Ls)
          trainer.step(batch.data[0].shape[0])
```

### class HybridBlock

file: python/mxnet/gluon/block.py

**HybridBlock.forward()** :

```python
def forward(self, x, *args):
    """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
            ......
            if isinstance(x, NDArray):
                params = {i: j.data(ctx) for i, j in self._reg_params.items()}
                except DeferredInitializationError:
                    self._deferred_infer_shape(x, *args)
                    for _, i in self.params.items():
                        i._finish_deferred_init()
                        params = {i: j.data(ctx) for i, j in self._reg_params.items()}

                        return self.hybrid_forward(ndarray, x, *args, **params)

            assert isinstance(x, Symbol), \
            "HybridBlock requires the first argument to forward be either " \
            "Symbol or NDArray, but got %s"%type(x)
            params = {i: j.var() for i, j in self._reg_params.items()}
            with self.name_scope():
                return self.hybrid_forward(symbol, x, *args, **params)
```



### class VGG

file: python/mxnet/gluon/model_zoo/vision/vgg.py

**VGG.hybrid_forward()**

```python
def hybrid_forward(self, F, x):
    x = self.features(x)
    x = self.output(x)
    return x
```
self.features 和 self.output 中有很多 **网络层**， 如     https://dgschwend.github.io/netscope/#/preset/vgg-16 中所示的各种层。 每一个层都是一个 HybridBlock 实例。hybrid_forward() 会根据这些层的add添加顺序以此执行层计算。 vgg16 所有层的添加定义如下。每一个HybridBlock 实例执行自身的self.\_\_call\_\_() 时都是执行 self.forward() -> self.hybrid_forward()

```python
def __init__(self, layers, filters, classes=1000, batch_norm=False, **kwargs):
          super(VGG, self).__init__(**kwargs)
          assert len(layers) == len(filters)
          with self.name_scope():
              self.features = self._make_features(layers, filters, batch_norm)
              self.features.add(nn.Dense(4096, activation='relu',
                                         weight_initializer='normal',
                                         bias_initializer='zeros'))
              self.features.add(nn.Dropout(rate=0.5))
              self.features.add(nn.Dense(4096, activation='relu',
                                         weight_initializer='normal',
                                         bias_initializer='zeros'))
              self.features.add(nn.Dropout(rate=0.5))
              self.output = nn.Dense(classes,
                                     weight_initializer='normal',
                                     bias_initializer='zeros')
  
      def _make_features(self, layers, filters, batch_norm):
          featurizer = nn.HybridSequential(prefix='')
          for i, num in enumerate(layers):
              for _ in range(num):
                  featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                           weight_initializer=Xavier(rnd_type='gaussian',
                                                                     factor_type='out',
                                                                     magnitude=2),
                                           bias_initializer='zeros'))
                  if batch_norm:
                      featurizer.add(nn.BatchNorm())
                  featurizer.add(nn.Activation('relu'))
              featurizer.add(nn.MaxPool2D(strides=2))
          return featurizer
```

### Dense and Cov2D

python class **nn.Dense** : python/mxnet/gluon/nn/basic_layers.py

python class **nn.Conv2D** : python/mxnet/gluon/nn/conv_layers.py







