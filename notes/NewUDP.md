# NewUDP(FPT)

## UDP packet processing time 

append() is main time-costing operation when processing a packet.  Consider append() = memcpy().

link bandwidth: 10GB

- 1)

  SArray.append (64KB)   :  35us  

  UDP.send(64KB) : 40us

  

  sendPacket():  70us

  ```c++
  sendPacket(){
  
  	SArray.append(64KB)   + UDP.send(64KB) 64KB
  
  }
  ```

  

  BW util: ~70%





- 2)

  Main thread :   use SArray.append (64KB)   to create packet, and push it into queue.

  Sending thread :  pop packet from queue, and run UDP.send(64KB) .

  The 2 threads affect each other's running time.

  

  BW util: 80-90%



- 3) 

  the sending thread use a unique CPU core?













