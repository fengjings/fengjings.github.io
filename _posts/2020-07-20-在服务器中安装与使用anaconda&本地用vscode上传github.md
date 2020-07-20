# 在服务器中安装与使用anaconda&本地用vscode上传github

## 在服务器中安装与使用anaconda
参考网站：  
[https://zhuanlan.zhihu.com/p/44398592](https://zhuanlan.zhihu.com/p/44398592)

准备工作：
+ 在个人电脑下载[mobaXterm](https://mobaxterm.mobatek.net/)并安装，选择session-ssh-输入remote host，管理员给的名字，第一次还需要输入密码（后续就记住了）。这一步是保证自己能登录上服务器。  
+ 在服务器中下载anaconda，我使用的服务器很容易中断，因此在本地下载好后上传上去。之后安装使用bash命令。  
```sh
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh
```

+ 按照帮助[tsinghua.help](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)修改.condarc文件，我发现服务器中并没有这个文件，因此首先创建它，然后将链接中那一段代码复制进去，如下所示。
```sh
conda config --add channels r
```

```sh
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```
+ 运行下列代码清除索引缓存，保证用的是镜像站提供的索引。
```sh
conda clean -i 
```
+ 然后可使用如下命令查看安装是否成功，显示conda版本即可。
```sh
conda -V
```
+ 用下列命令去创建环境，由于服务器有些库下不下来，需要找找原因
```sh
conda create -n myenv numpy
or
conda create --name py35 python=3.5
```
+ 上一步中，create不了，发现是源的问题，因此进行了换源，可以运行上面那句话了。查看安装环境可以通过：
```sh
conda info --env
or
conda info -e
```
+ 激活与取消激活环境可以通过：
```sh
conda activate myenv
conda deactivate
```


## 本地用vscode上传github

+ 下载[vscode](https://code.visualstudio.com/)并安装。  
+ 下载[git](https://git-scm.com/)并安装（在其中一步选择vscode作为打开方式）。
+ 在某个路径下，从[github](https://github.com/)上下一个repo，如果没有github则创建账号并在github上新建一个repo并下载。
```sh
git clone "github path"
```
+ 用vscode打开那个repo的文件夹，则能看到所有文件。
+ 编辑一个markdown文件，可通过control（mac中按command） + k 然后按 v 来实时预览markdown文件。  
+ 在vscode中使用terminal并尝试git add没有问题，git commit需要输入global.email和global.name，输入完毕后即可使用，git push origin master则需要输入账号和密码。之后便可以使用vscode进行修改与上传。
+ 在vscode左侧，有source control，点进去便可以看到修改，以及可以进行commit和push，而不用到terminal中进行。对于修改的文件，可以点击‘+’得到stage changes，以及上方有commit之类的操作。