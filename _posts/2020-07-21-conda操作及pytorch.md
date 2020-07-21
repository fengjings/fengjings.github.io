# Conda 新建环境以及删除
+ 新建pytorch环境方法(已经换为国内源)以及相关命令
```sh
新建环境
conda create -n torch-py37 python=3.7
查看环境
conda info --envs   or conda info -e or conda env list
安装pytorch
conda install pytorch torchvision cudatoolkit=9.2
nvcc -V
激活
conda activate torch-py37
取消激活
conda deactivate
删除虚拟环境：
conda remove -n your_env_name --all
or
conda remove --name your_env_name --all
```