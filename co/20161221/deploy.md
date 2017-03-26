#安装Python的步骤
mkdir -p /home/limeihang/lcy/software/python27
./configure --prefix="/home/limeihang/lcy/software/python27"
/home/limeihang/lcy/software/python27/bin/python setup.py install
/home/limeihang/lcy/software/python27/bin/python setup.py install
#到达制定目录
cd /home/limeihang/lcy/software/python27/bin/
#设置Python为编译器
PATH=$PATH:/home/limeihang/lcy/software/python27/bin
export PATH=/home/limeihang/lcy/software/python27/bin/:$PATH
source .bash_profile
#查看运行结果
ps -ef |grep main2.py
#运行程序
nohup python main3.py &