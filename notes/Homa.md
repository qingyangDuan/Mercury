# Homa （DPDK）

## 1 install Homa and DPDK

看对应的 DPDK 库的 Makefile.txt 和 Homa project的 CMakeFiles.txt 文件。DPDK 库把其 name.a 库文件安装在 /usr/local/lib 中，把对应的 name.h 头文件安装在 /usr/local/include 中, 以及一些可执行文件如testpmd在/usr/local/bin中。 Homa project 使用 DPDK 的这些库文件，最后生成 homa.a 库文件供上层应用使用。



Homa 代码： https://github.com/PlatformLab/Homa。 其中基类 Driver 为一个抽象类，不可实例化。 其两个子类 DpdkDriver 和 FakeDriver 为 具体的实现。这两个子类使用上述的 DPDK 库的功能，实现了不可靠的数据包传输。 两个子类很相似，多定义一个 FakeDriver 的目的暂时还没弄清。 Homa::Transport 类基于上述两个子类之一（使用基类指针 Driver * 调用子类的函数功能）的不可靠数据包传输，实现Homa的传输协议。 应用程序通过实例化 Transport 类来使用 Homa。 

### enable IOMMU (required for vfio-pci) and Hugepages

```shell
sudo vim /etc/default/grub
#add GRUB_CMDLINE_LINUX="iommu=on intel_iommu=on default_hugepagesz=1G hugepagesz=1G hugepages=4"
sudo update-grub
sudo reboot
sudo dmesg | grep IOMMU   
cat /proc/meminfo | grep Huge
```
### install MLNX-OFED
```shell
MLNX_OFED_VER=4.9-2.2.4.0
OS_VER="ubuntu`lsb_release -r | cut -d":" -f2 | xargs`"
MLNX_OFED="MLNX_OFED_LINUX-$MLNX_OFED_VER-$OS_VER-x86_64"
wget http://www.mellanox.com/downloads/ofed/MLNX_OFED-$MLNX_OFED_VER/$MLNX_OFED.tgz
tar xzf $MLNX_OFED.tgz
sudo ./$MLNX_OFED/mlnxofedinstall --force --without-fw-update
sudo /etc/init.d/openibd restart
# reboot

#check ofed
sudo mst start
sudo mlxfwmanager
```

### install DPDK

```shell
wget http://fast.dpdk.org/rel/dpdk-18.11.11.tar.xz
tar xf dpdk-18.11.11.tar.xz
mv dpdk-stable-18.11.11 dpdk
cd dpdk
sudo apt install libnuma-dev
make defconfig
make
sudo make install
export RTE_SDK=$PWD
export RTE_TARGET=build
make -C examples
./usertools/dpdk-setup.sh ##bind to vfio-pci
```

### uninstall DPDK

```
sudo rm /usr/local/lib/librte_*
sudo rm /usr/local/lib/libdpdk.a
sudo rm -r /usr/local/share/dpdk
sudo rm -r /usr/local/include/dpdk
sudo rm /usr/local/bin/dpdk*
sudo rm /usr/local/bin/test*
```



DPDK will be installed in /usr/local/lib  && /usr/local/bin

DPDK doc: 

https://doc.dpdk.org/guides-18.11/linux_gsg/sys_reqs.html



use `ifconfig  ....... down` to stop nic interface, then bind it to igb_uio driver





## 2 arch

![](Homa-uml.png)

- 应用调用Transport 类（ 具体实现为TransportImpl类 ）来传输和接收数据（主要通过图中的 alloc() receive() poll() 三个函数）。

- Homa 的包调度、流调度、可靠传输等主要算法逻辑在Sender 和Receiver类中实现。这两个类每个大概一千多行代码，包括几个主要的对到来的各种包的处理函数，以及Timer、packetQueue、messageQueue等操作。

- 中间位置的 DpdkDriver 为 dpdk驱动的实现。FakeDriver 实现了一个简单的模拟网络，不需要dpdk库和网卡，其目的是对Homa主要功能的测试和debug。

## 3 classes







