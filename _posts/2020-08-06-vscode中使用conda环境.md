# vscode使用conda环境

前文已经叙述过conda相关操作。  
 
本文研究如何在vscode中使用anaconda中的环境。  

下文相关操作参考了网站：  
<https://blog.csdn.net/Add_a_cat/article/details/101051759>

## important note

安装anaconda时应选择所有用户（all users）,如果选择当前用户，则jupyter notebook会有问题。

同时，可不用选择添加到系统环境。

## conda相关操作，jupyter notebook相关操作

+ 安装  

jupyter中添加conda环境： https://www.cnblogs.com/hgl0417/p/8204221.html
```sh
conda info -e
conda create -n py37torch python=3.7 ipykernel
conda info -e
conda activate py37torch
nvcc -V
conda activate py37torchconda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install jupyter
python -m ipykernel install --user --name 37torch --display-name "37-torch"
jupyter notebook即可选择kernel
```


## 下载并安装vscode

相关操作简单，不赘述。

## 修改setting.json

ctrl+shift+p 快捷键，搜索setting.json  
在其中进行如下修改：
```sh
{
    ......
    "python.pythonPath": "C:\\ProgramData\\Anaconda3\\python.exe",
    "terminal.integrated.shell.windows": "C:\\Windows\\System32\\cmd.exe",
    "terminal.integrated.shellArgs.windows": [
        "/K",
        "C:\\ProgramData\\Anaconda3\\Scripts\\activate.bat C:\\ProgramData\\Anaconda3"
    ]
}
```
`python.pythonPath` 添加之后，debug会直接调用这个python.exe，
在终端运行时，也会自动先运行 conda active 环境名

另外两个配置是关键，在设置里搜索shell 会找到terminal.integrated.shell.windows这个配置项

找到Anaconda Prompt，右键复制出其路径。 

`"C:\\Windows\\System32\\cmd.exe"`  
`"/K"`  
`"C:\\Users\\FengJing_W\\anaconda3\\Scripts\\activate.bat C:\\Users\\FengJing_W\\anaconda3"`  

第一段是终端窗口，这里建议就用 cmd 不要用 powershell ，因为 powershell 对虚拟终端的支持有些问题。

第二三段复制到terminal.integrated.shellArgs.windows下，对 \ 符号做转义。

保存重启就能看到效果了。

```sh
(base) F:\github\fengjings.github.io>conda info -e
# conda environments:
#
base                  *  C:\Users\FengJing_W\anaconda3
py-torch                 C:\Users\FengJing_W\anaconda3\envs\py-torch


(base) F:\github\fengjings.github.io>conda activate py-torch

(py-torch) F:\github\fengjings.github.io>   
```


