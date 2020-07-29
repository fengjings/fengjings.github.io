# 在本地使用服务器的jupyter notebook

前文已经叙述过服务器的操作。  
目前已经可以运行GPU版的py程序。  
本文研究如何在本地使用服务器的jupyter notebook（已经由anaconda安装过）  
判断是否有jupyter notebook，在服务器命令行中输入下面语句，不报错即有：  
```shell
jupyter notebook
```
参考网站：  
<http://www.bubuko.com/infodetail-3465538.html>
## 服务器端配置jupyter notebook

+ 创建配置文件  
默认情况下，配置文件 ~/.jupyter/jupyter_notebook_config.py 并不存在  
需要自行创建。  
运行以下命令创建：
```shell
jupyter notebook --generate-config
执行成功后提示以下信息：
Writing default config to: /home/username/.jupyter/jupyter_notebook_config.py
```

+ 生成密码  
服务器端命令行输入:  
```shell
jupyter notebook password
Enter password: 
Verify password: 
[NotebookPasswordApp] Wrote hashed password to /home/username/.jupyter/jupyter_notebook_config.json
```
此时会提示输入密码及确认密码，密码设置完成后提示将生成的密码写入/home/username/.jupyter/jupyter_notebook_config.json，注意username视用户而定，会直接出现在提示信息中。

打开存储密码的json文件，可以看到：
```shell
"password": "sha1:*************:******************"
```
复制"sha1......"密文

+ 修改配置文件  
在/home/username/.jupyter/jupyter_notebook_config.py中找到以下行，修改为：
```shell
c.NotebookApp.ip='*' #允许访问的IP地址，设置为*代表允许任何客户端访问
c.NotebookApp.password = u'sha1:...刚才生成密码时复制的密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =9999 #可自行指定一个端口, 访问时使用该端口，默认8888，修改为9999
c.NotebookApp.allow_remote_access = True
```

## 服务器端启动jupyter notebook  
```shell
jupyter notebook
```
显示
```shell
[I 10:02:48.850 NotebookApp] The Jupyter Notebook is running at:
[I 10:02:48.850 NotebookApp] http://ubuntu:9999/
```
在本地服务器的chrome中打开
```shell
http://服务器端口:9999/
```
显示如下即成功，然后即可如本地jupyter notebook相同使用：
```shell
[I 10:03:48.257 NotebookApp] 302 GET / ([ip]) 1.06ms
[I 10:03:48.263 NotebookApp] 302 GET /tree? ([ip]) 0.75ms
[I 10:03:52.761 NotebookApp] 302 POST /login?next=%2Ftree%3F ([ip]) 1.49ms
```



