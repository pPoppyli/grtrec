import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F     # 激励函数都在这

import jieba
from numpy import *
import os
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import CountVectorizer   #词频矩阵

def readFile(path):
    with open(path, 'r', errors='ignore',encoding='gbk') as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        file.close()
        return content
 
 
def saveFile(path, result):
    with open(path, 'w', errors='ignore',encoding='gbk') as file:
        file.write(result)
        file.close()
 
 
def segText(inputPath):
    data_list = []
    label_list = []
    fatherLists = os.listdir(inputPath)  # 主目录
    for eachDir in fatherLists:  # 遍历主目录中各个文件夹
        eachPath = inputPath +"/"+ eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
        for eachFile in childLists:  # 遍历每个文件夹中的子文件
            eachPathFile = eachPath + eachFile  # 获得每个文件路径
            content = readFile(eachPathFile)  # 调用上面函数读取内容
            result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
            cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            label_list.append(eachDir)
            data_list.append(" ".join(cutResult))
    return  data_list,label_list
 
def getStopWord(inputFile):
    stopWordList = readFile(inputFile).splitlines()
    return stopWordList
 

def getTFIDFMat(train_data,train_label, stopWordList):  # 求得TF-IDF向量
    class0 = ''
    class1 = ''
    for num in range(len(train_label)):
        if train_label[num]=='绿茶':
            class0 = class0 + train_data[num]
            train_label[num] = 1
        elif train_label[num]=='朋友':
            class1 = class1 + train_data[num]
            train_label[num] = 0
    train = [class0,class1]
    vectorizer = CountVectorizer(stop_words=stopWordList, min_df=0.5)  # 其他类别专用分类，该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    cipin = vectorizer.fit_transform(train)

    tfidf = transformer.fit_transform(cipin)  # if-idf中的输入为已经处理过的词频矩阵
    
    train_cipin = vectorizer.transform(train_data)
    train_arr = transformer.transform(train_cipin)
    train_arr = torch.Tensor(train_arr.toarray())
    print(train_cipin)
    train_label = torch.LongTensor(train_label)
    return train_arr,train_label,train


class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x

    def predict(self,x,train):
        data_list = []
        stopWordList = getStopWord('stop/stopword.txt')  
        vectorizer = CountVectorizer(stop_words=stopWordList, min_df=0.5)

        transformer = TfidfTransformer()  
        cipin = vectorizer.fit_transform(train)
        tfidf = transformer.fit_transform(cipin)
        result = x.replace("\r\n", "").strip()  # 删除多余空行与空格
        cutResult = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
        data_list.append(" ".join(cutResult))
        x_cipin = vectorizer.transform(data_list)
        x_arr = transformer.transform(x_cipin)
        x_arr = torch.Tensor(x_arr.toarray())
        y_pre = self.forward(x_arr)
        y = torch.ones(1)
        y=y.long()
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(y_pre,y)
        # print(loss)
        if loss<0.5:
            # print("绿茶")
            return 1
        else:
            # print("朋友")
            return 0
       
