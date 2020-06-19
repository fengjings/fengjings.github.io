****
# 如何安装多版本cuda并切换

### introduction
继续用昨天的电脑进行操作

查看当前cuda版本
```sh
nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
```
下载其他版本  
最新的在这里，目前是cuda11 <https://developer.nvidia.com/cuda-downloads>  
我想使用10.2, <https://developer.nvidia.com/cuda-10.0-download-archive>  
使用run版本，下载并安装   
```sh
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
```
中间会有些步骤，我把其中显卡驱动取消了，不然过不去。
路径在，安装完可以在这里看到三个：
```sh
/usr/local/
cuda-9.0 
cuda-10.2
cuda
```
修改bash中的环境，其中搜索cuda可看到9.0,由于有软链接，将9.0删除即可。          
```sh
~/.bashrc
sudo gedit ~/.bashrc
#################################
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
....
###################################
source ~/.bashrc
```

使用的时候修改软连接
```sh
sudo rm -rf cuda
sudo ln -s /usr/local/cuda-9.0/  /usr/local/cuda
nvcc -V
```

goto <https://pytorch.org/get-started/locally/> to find the proper command  
conda命令  
```sh
conda create -n torch-py37 python=3.7
conda info --envs
fengjing@fengjing-desktop:~$ source activate torch-py37
(torch-py37) fengjing@fengjing-desktop:~$ conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

```


proxy command  
```sh
"Unable to determine SOCKS version from %s" % proxy_url
    ValueError: Unable to determine SOCKS version from socks://127.0.0.1:7891/
unset all_proxy && unset ALL_PROXY
export all_proxy="socks5://127.0.0.1:7891"
export all_proxy="https://127.0.0.1:7890"
export all_proxy="http://127.0.0.1:7890"
```









