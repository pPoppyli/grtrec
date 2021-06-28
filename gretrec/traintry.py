from model import Net,segText,getTFIDFMat,getStopWord
from torch.autograd import Variable
import torch
import os
import time
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import CountVectorizer   #词频矩阵

def readFile(path):
    with open(path, 'r', errors='ignore',encoding='gbk') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        file.close()
        return content
# import Run
# run = Run.get_context()
net = Net(n_feature=1922, n_hidden=1, n_output=2) # 几个类别就几个 output
# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.CrossEntropyLoss()
data,label = segText('data')
stopWordList = getStopWord('stop/stopword.txt')  # 获取停用词表
x,y,z = getTFIDFMat(train_data=data,train_label=label,stopWordList=stopWordList)
x, y = Variable(x), Variable(y)

start =time.perf_counter()  
for t in range(5000):
    
    out = net(x)     # 喂给 net 训练数据 x, 输出分析值
    loss = loss_func(out, y)     # 计算两者的误差
    
    # run.log('loss', loss)
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    # if t%500==0:
    #     print(loss.item())


end = time.perf_counter()
print('Running time: %s Seconds'%(end-start))

# while 1:
#     print('请输入需要预测的文本:')
#     a = input()
#     net.predict(a,z)



prelist=[]
prelist2=[]
acc = 0
fatherLists = os.listdir("./testdata/绿茶")  # 主目录
for eachDir in fatherLists:  # 遍历主目录中各个文件夹
    a = readFile("./testdata/绿茶"+'//'+eachDir)
    prd = net.predict(a,z)
    prelist.append(prd)
        # print(prelist)
for i in prelist:
    if i==1:
        acc = acc+1

Acc = acc/100
print(Acc)