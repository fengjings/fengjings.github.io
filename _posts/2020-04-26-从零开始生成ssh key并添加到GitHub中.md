****
# 从零开始生成ssh key并添加到GitHub中

### introduction
换了台电脑，之前的操作都忘了，重新走了一遍，记录一下，Win10系统。

### SSH  
`<description link>` : <https://en.wikipedia.org/wiki/Secure_Shell>  
**Secure Shell**（安全外壳协议，简称**SSH**）是一种加密的网络传输协议，可在不安全的网络中为网络服务提供安全的传输环境。

### 注册GitHub账号
`<official link>` : <https://github.com>  

### 下载bash  
Bash是一个命令处理器，通常运行于文本窗口中，并能执行用户直接输入的命令。
`<description link>` : <https://en.wikipedia.org/wiki/Bash_(Unix_shell)>  
`<download link>` : <https://www.gnu.org/software/bash/>  


### 打开bash窗口    
在桌面点击右键，选择其中“Git Bash Here”，即可以打开bash窗口。
输入 (将example邮箱替换为GitHub的注册邮箱)：
```sh
$ ssh-keygen -t rsa -C "example@example.com"
```

回车即可，不需要改内容，连续三次回车：
```sh
Generating public/private rsa key pair.
Enter file in which to save the key (/c/Users/YOURNAME/.ssh/id_rsa):
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
```
之后会出现：
```sh
Your identification has been saved in /c/Users/YOURNAME/.ssh/id_rsa.
Your public key has been saved in /c/Users/YOURNAME/.ssh/id_rsa.pub.
The key fingerprint is:
............
```

然后本地就没有问题了，可以在C盘->user->'Yourname'->.ssh文件夹中找到上面两个文件。
如果找不到这个文件夹，看看是否文件夹被隐藏了。

### 将key添加到github中  
打开刚才生成的.pub文件（在.ssh文件夹中），注意是ssh-rsa开头的一串数。  
将其复制。（也可用vim等）  
打开GitHub，将鼠标移至右上角的头像，打开列表中的**Setting**  
在左侧列表中选取**SSH and GPG keys**
注意到上半部分是**SSH keys**，点击右侧**New SSH key**  
**Title**可自定  
将刚才复制的.pub文件内容粘贴在**key**处    
    
即可  

### 测试  
在Bash中输入：
```sh
$ ssh git@github.com
```
出现：
```sh
Warning: Permanently added the RSA host key for IP address 'address' to the list of known hosts.
PTY allocation request failed on channel 0
Hi 'NAME'! You have successfully authenticated, but GitHub does not provide shell access.
Connection to github.com closed.
```

表明成功。

下一步即可进行clone或者push等操作。  

```sh
$ git clone git@github.com..........
$ git status
$ git push
......
```




