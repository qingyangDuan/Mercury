# 1) work for all users
add ld_path to /etc/ld.so.conf    
then run `sudo ldconfig`

# 2) only work for my user
modify `LD_LIBRARY_PATH` in  `~/.bashrc`    
then run `source .bashrc`
