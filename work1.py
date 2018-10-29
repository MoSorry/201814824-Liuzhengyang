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
originSample = 'D:/data mining/20news-18828' #确定数据存储的目录
processedSample = 'D:/data mining/20news'     #新的存放数据的文件夹

def createFiles():                                #将数据复制到新文件夹，避免原始数据破坏
    srcFilesList = listdir('originSample')      #将目录下每个文件夹名称以组的形式保存
    for i in range(len(srcFilesList)):
        if i==0: continue
        dataFilesDir = 'originSample/' + srcFilesList[i]  # 确定每个文件夹的路径
        dataFilesList = listdir(dataFilesDir)               #确定每个新闻类别路径下的具体文件名称
        targetDir = 'processedSample/' + srcFilesList[i] # 新文件夹的每个的路径
        if path.exists(targetDir)==False:
            mkdir(targetDir)                                #创建相应的文件夹
        for j in range(len(dataFilesList)):
            createProcessFile(srcFilesList[i],dataFilesList[j])#调用createProcessFile()在新文档中处理文本

#在前期处理中因为anaconda中并没有textblob的库，所以借鉴网上的方法，将数据重新复制排版代替分词操作。

def createProcessFile(srcFilesName,dataFilesName):
    srcFile = 'originSample/' + srcFilesName + '/' + dataFilesName
    targetFile= 'processedSample/' + srcFilesName\
                + '/' + dataFilesName
    fw = open(targetFile,'w')
    dataList = open(srcFile).readlines()
    for line in dataList:          #复制新闻内容
        resLine = lineProcess(line) # 调用lineProcess()处理每行文本
        for word in resLine:
            fw.write('%s\n' % word) #一行一个单词
    fw.close()
#######################################################预处理###########################################################
def lineProcess(line):
    stopwords = nltk.corpus.stopwords.words('english') #去停用词
    porter = nltk.PorterStemmer()  #词干分析
    splitter = re.compile('[^a-zA-Z]')  #去除非字母字符，形成分隔
    words = [porter.stem(word.lower()) for word in splitter.split(line)\
             if len(word)>0 and\
             word.lower() not in stopwords]
    return words


##################################################### IDF###############################################################
def computeIDF():
    fileDir = 'processedSample'
    wordDocMap = {}  # <word, (docM,...,docN)>
    IDFPerWord = {}  # <word, IDF>
    countDoc = 0.0
    cateList = listdir(fileDir)
    for i in range(len(cateList)):
        sampleDir = fileDir + '/' + cateList[i]
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
        IDF = log(20000 / countDoc)
        IDFPerWord[word] = IDF
    return IDFPerWord

def main():
    start=time.clock()
    IDFPerWord = computeIDF()
    end=time.clock()
    print ('runtime: ' + str(end-start))
    fw = open('IDFPerWord','w')
    for word, IDF in IDFPerWord.items():
        fw.write('%s %.6f\n' % (word,IDF))
    fw.close()