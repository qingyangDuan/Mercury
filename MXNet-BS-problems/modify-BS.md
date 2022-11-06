# Install BS from source
- pre-requisite: 
   sudo python3 -m pip install bayesian-optimization==1.0.1 six
- cd bytescheduler/
- vim setup.py to add MXNET_ROOT path
- python3 setup.py install --user
- One problem: `cannot find \**.so`   
This is because LD_LIBRARY_PATH in sudo is different with that in local user.  
If we want it to be global, we need to add lib path to /etc/ld.so.conf then run sudo ldconfig.  
Or, just add this lib path to LD_LIBRARY_PATH in ~/.bashrc.

# Modify BS python lib:
- python -m pip uninstall bytescheduler   
- change BS python code in ~/Bytescheduler  
- python3 -m pip setup.py install --user   // This will update BS lib python in ~/.local/lib/python(2 or 3).


