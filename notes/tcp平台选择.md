

# user space TCP



| 选择              | 基于DPDK | 拥塞控制算法                           | L4 TCP性能 | 实现难点               | 缺点                   |
| ----------------- | -------- | -------------------------------------- | ---------- | ---------------------- | ---------------------- |
| UDT               |          | udt-cc                                 |            |                        | 不能跑满10GB，cc性能差 |
| TLDK              | 是       | newreno                                |            |                        |                        |
| f-stack           | 是       | cubic，dctcp（user space FreeBSD-TCP） |            |                        |                        |
| **libVMA**        |          | newreno-lwip，cubic                    |            | 如何增加新的socket api |                        |
| **VPP-HostStack** | 是       | newreno，cubic                         |            | 更改tcp难吗？          |                        |
| OFP               |          | newreno                                |            |                        |                        |
| DPDK-ANS          | 是       |                                        |            |                        | 未开源                 |
| mTCP              | 是       | newreno                                |            |                        |                        |
| seaStar           | 是       |                                        |            |                        |                        |
| SandStorm         |          |                                        |            |                        |                        |