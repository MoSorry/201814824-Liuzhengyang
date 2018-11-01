import nltk
import os
import chardet
import string
import collections
import numpy
import json
import time
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from tkinter import _flatten
from numpy import *
from os import listdir,mkdir,path
import re
import operator


def createFiles(oripath,path):                                #将数据复制到新文件夹，避免原始数据破坏
    FList = listdir('oripath')             #将目录下每个文件夹名称以组的形式保存
    for i in range(len(FList)):
        if i==0: continue
        dataFilesDir = 'oripath/' + FList[i]
        dataFilesList = listdir(dataFilesDir)               #确定每个新闻类别路径下的具体文件名称
        targetDir = 'path/' + FList[i]
        if path.exists(targetDir)==False:
            mkdir(targetDir)
        for j in range(len(dataFilesList)):
            createProcessFile(FList[i],dataFilesList[j])

def createProcessFile(srcFilesName,dataFilesName):
    srcFile = 'oripath/' + srcFilesName + '/' + dataFilesName
    targetFile= 'path/' + srcFilesName\
                + '/' + dataFilesName
    fw = open(targetFile,'w')
    dataList = open(srcFile).readlines()
    for line in dataList:          #复制新闻内容
        resLine = lineProcess(line)
        for word in resLine:
            fw.write('%s\n' % word) #一行一个单词
    fw.close()
######################################词干提取##################################
def steming(docwordlist):  # 词干提取
        st_wordlist = []
        stemmer = SnowballStemmer("english")  # 选择一种语言
        for each in docwordlist:
            st_wordlist.append(stemmer.stem(each))
        return st_wordlist
##########################预处理###########################################################
def lineProcess(line):
    stopwords = nltk.corpus.stopwords.words('english') #去停用词
    porter = nltk.PorterStemmer()  #词干分析
    splitter = re.compile('[^a-zA-Z]')  #去除非字母字符，形成分隔
    words = [porter.stem(word.lower()) for word in splitter.split(line)\
             if len(word)>0 and\
             word.lower() not in stopwords]
    return words

def countWords(path):
    wordMap = {}
    newWordMap = {}
    sampleFilesList = listdir(path)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = path + '/' + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        for j in range(len(sampleList)):
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir).readlines():
                word = line.strip('\n')
                wordMap[word] = wordMap.get(word,0.0) + 1.0
    for key, value in wordMap.items():
        if value > 4:
            newWordMap[key] = value
    sortedNewWordMap = sorted(newWordMap.iteritems())
    print ('wordMap size : %d' % len(wordMap))
    return sortedNewWordMap

def printWordMap():
    countLine=0
    fr = open('D:\\Vector\\WordCountMap.txt','w')
    sortedWordMap = countWords()
    for item in sortedWordMap:
        fr.write('%s %.1f\n' % (item[0],item[1]))
        countLine += 1
##################################################### IDF###############################################################
def computeIDF():
    fileDir = 'sample'
    wordDocMap = {}  # <word, (docM,...,docN)>
    IDFPerWord = {}  # <word, IDF>
    countDoc = 0.0
    cList = listdir(fileDir)
    for i in range(len(cList)):
        sampleDir = fileDir + '/' + cList[i]
        sampleList = listdir(sampleDir)
        for j in range(len(sampleList)):
            sample = sampleDir + '/' + sampleList[j]
            for line in open(sample).readlines():
                word = line.strip('\n')
                if word in wordDocMap.keys():
                    wordDocMap[word].add(sampleList[j])  # 保存单词出现过的文档
                else:
                    wordDocMap.setdefault(word, set())
                    wordDocMap[word].add(sampleList[j])

    for word in wordDocMap.keys():
        countDoc = len(wordDocMap[word])  # 统计文档个数
        IDF = log(20000 / countDoc)/log(10)
        IDFPerWord[word] = IDF
    return IDFPerWord


######################################################TF###################################
def computeTF(indexOfSample, trainSamplePercent):
        IDFPerWord = {}  # <word, IDF值> 从文件中读入后的数据保存在此字典结构中
        for line in open('IDFPerWord').readlines():
            (word, IDF) = line.strip('\n').split(' ')
            IDFPerWord[word] = IDF

        fileDir = 'path'
        trainFileDir = "Vector/" + 'TFIDFMap' + str(indexOfSample)
        testFileDir = "Vector/" + 'TFIDFMap' + str(indexOfSample)

        tsTrainWriter = open(trainFileDir, 'w')
        tsTestWriter = open(testFileDir, 'w')

        cateList = listdir(fileDir)
        for i in range(len(cateList)):
            sampleDir = fileDir + '/' + cateList[i]
            sampleList = listdir(sampleDir)

            testBeginIndex = indexOfSample * (len(sampleList) * (1 - trainSamplePercent))
            testEndIndex = (indexOfSample + 1) * (len(sampleList) * (1 - trainSamplePercent))

            for j in range(len(sampleList)):
                TFPerDocMap = {}  # <word, 文档doc下该word的出现次数>
                sumPerDoc = 0  # 记录文档doc下的单词总数
                sample = sampleDir + '/' + sampleList[j]
                for line in open(sample).readlines():
                    sumPerDoc += 1
                    word = line.strip('\n')
                    TFPerDocMap[word] = TFPerDocMap.get(word, 0) + 1

                if (j >= testBeginIndex) and (j <= testEndIndex):
                    tsWriter = tsTestWriter
                else:
                    tsWriter = tsTrainWriter

                tsWriter.write('%s %s ' % (cateList[i], sampleList[j]))  # 写入类别cate，文档doc

                for word, count in TFPerDocMap.items():
                    TF = float(count) / float(sumPerDoc)
                    tsWriter.write('%s %f ' % (word, TF * float(IDFPerWord[word])))  # 继续写入类别cate下文档doc下的所有单词及它的TF-IDF值

                tsWriter.write('\n')
        tsTrainWriter.close()
        tsTestWriter.close()
        tsWriter.close()

def CreateVSM(path):
            database = list(_flatten(path))
            Vocabulary = printWordMap()
            Document_Frequency =countWords(path)
            IDF_v = computeIDF()
            TF_list = []
            VSM = []
            for Document in Document_Frequency:
                TF_v = computeTF(Document, Vocabulary)
                TF_list.append(TF_v)
                VSM_v = TF_v * IDF_v
                print(IDF_v, TF_v, VSM_v)
                VSM.append(VSM_v)

            return VSM

def GetData(path):
    VSM = numpy.fromfile(path)
    VSM = VSM.reshape(18828,len(VSM)//18828)
    print(VSM)
    return VSM
def GetLabel(path):
    file = open(path,'rb')
    label = json.load(file)
    return numpy.array(label)


def DataDivide(label,VSM):
    data = numpy.c_[VSM,label]
    label_list = list(label)
    con = collections.Counter(label_list)
    pro = list(dict(sorted(dict(con.most_common()).items(),key = lambda k:k[0])).values())
    start = [0]
    cal = 0
    for i in range(len(pro)):
        cal += pro[i]
        start.append(cal)
    print(start)
    test_pos = list(numpy.trunc(numpy.array(pro) * 0.2))
    test_range = []
    train_range = []
    for i in range(20):
        a = []
        a.append(start[i])
        a.append(int(test_pos[i]+start[i])+1)
        test_range.append(a)
        b = []
        b.append(int(test_pos[i]+start[i])+1)
        b.append(start[i+1])
        train_range.append(b)
    test_data = data[test_range[0][0]:test_range[0][1],:]
    print(len(test_data))
    train_data = data[train_range[0][0]:train_range[0][1],:]

    for i in range(19):
        c = data[test_range[i+1][0]:test_range[i+1][1],:]
        d = data[train_range[i+1][0]:train_range[i+1][1],:]
        test_data = numpy.r_[test_data,c]
        train_data = numpy.r_[train_data,d]
    train_data.tofile(r"E:\data mining\train_data.txt")
    test_data.tofile(r"E:\data mining\test_data.txt")
    return train_data,test_data
def Dividexy(data,pos):
    x_data = data[:,0:pos]
    y_data = data[:,pos:pos+1]
    return x_data,y_data

##########################################KNN###################################################
def Disdence(x_train,x_test):
    print("----------------------计算距离------------------")
    inner_product = numpy.dot(x_train, x_test.T)
    print(inner_product.shape)
    norm_train = numpy.linalg.norm(x_train, ord=2, axis=1, keepdims=False)
    norm_test = numpy.linalg.norm(x_test, ord=2, axis=1, keepdims=False)
    norm_nd = numpy.dot(numpy.array([norm_train]).T, numpy.array([norm_test]))
    distance = inner_product / norm_nd
    print(distance.shape)
    print("-----------------距离计算结束--------------------")
    distance.tofile('E:\data mining\distance.txt')
    return numpy.nan_to_num(distance)

def KNN(distance,y_train,K):
    print("--------------KNN分类----------------")
    num_test = len(distance[0])#获得测试数据数量
    y_predict = []#预测类别列表
    for i in range(num_test):
        a = y_train[distance[:,i].argsort()]
        a = a.reshape(len(a), )
        a = a.tolist()
        a = list(reversed(a))
        NN = a[0:K:]
        class_bag = collections.Counter(NN)
        predict_label = class_bag.most_common(1)
        y_predict.append(list(dict(predict_label).keys())[0])
    print('----------------KNN分类结束-----------------------')
    return numpy.array(y_predict).reshape(len(y_predict),1)


def evaule(y_predict,y_test):
    M = len(y_predict)
    a = y_predict - y_test
    a = a.reshape(len(a), )
    b = collections.Counter(a.tolist())
    return b[0.0]/M


def main():
            oripath = 'D:/data mining/20news-18828'  # 原始目录
            path = 'D:/data mining/20news'
            createFiles(oripath,path)
            IDFPerWord = computeIDF()
            voc = 'path/' + 'vocabulary'
            mkdir(voc)
            fw = open('voc', 'w')
            for word, IDF in IDFPerWord.items():
                fw.write('%s %.6f\n' % (word, IDF))
            fw.close()

            VSM = CreateVSM(path)
            path_VSM = 'D:/data mining/VSM.txt'
            path_Label = 'D:/data mining/label.txt'

            VSM = GetData(path_VSM)
            Label = GetLabel(path_Label)
            train_data, test_data = DataDivide(Label, VSM)
            data = numpy.c_[VSM,Label]
            vec_length = len(VSM[0])
            x_train, y_train = Dividexy(train_data, vec_length)
            x_test, y_test = Dividexy(test_data, vec_length)
            distance = Disdence(x_train,x_test)
            y_predict = KNN(distance,y_train,11)
            y_predict.tofile(r"E:\data mining\predict.txt")
            print('the accurency is %f'%evaule(y_predict,y_test))
