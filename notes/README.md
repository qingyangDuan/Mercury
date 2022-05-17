# notes

These notes present details of how communication happens in DML(distributed machine learning) with MXNet.

MXNet adpot ps-lite as its PS architechture library. 

And ps-lite  utilize ZeroMQ to send data. 

These notes show all the details of how gradients and parameters are sychronizad.
Data goes form the toppest-level training code (writen by users with python API) all the way down to the underlying ZMQ.
There are many layers of function calls, many python classes and many c++ classes.
You can find how they work together in my notes.

Some of the notes are in Chinese, sorry for that. I'm working to translate all of them to English. 
