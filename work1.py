import nltk
import os
import chardet
import string
import collections
import numpy
import json
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from tkinter import _flatten
from numpy import *
from os import listdir,mkdir,path
import re
import operator

originSample = r'D:\data mining\20news-18828' #确定数据存储的目录
processedSample = 'D:\data mining\20newsgroup' #新的存放数据的文件夹
def createFiles(): #将数据复制到新文件夹，避免原始数据破坏
    srcFilesList = listdir('originSample') #将目录下每个文件夹名称以组的形式保存
    for i in range(len(srcFilesList)):
        if i==0: continue
        dataFilesDir = 'originSample/' + srcFilesList[i] # 确定每个文件夹的路径
        dataFilesList = listdir(dataFilesDir) #确定每个新闻类别路径下的具体文件名称
        targetDir = 'processedSample/' + srcFilesList[i] # 新文件夹的每个的路径
        if path.exists(targetDir)==False:
            mkdir(targetDir)  #创建相应的文件夹
        for j in range(len(dataFilesList)):
            createProcessFile(srcFilesList[i],dataFilesList[j]) # 调用createProcessFile()在新文档中处理文本

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
#####################################################################预处理#############################################
def lineProcess(line):
    stopwords = nltk.corpus.stopwords.words('english') #去停用词
    porter = nltk.PorterStemmer()  #词干分析
    splitter = re.compile('[^a-zA-Z]')  #去除非字母字符，形成分隔
    words = [porter.stem(word.lower()) for word in splitter.split(line)\
             if len(word)>0 and\
             word.lower() not in stopwords]
    return words

########################################################统计每个词的出现次数############################################
def countWords():
    wordMap = {}
    newWordMap = {}
    fileDir = 'processedSample'
    sampleFilesList = listdir(fileDir)
    for i in range(len(sampleFilesList)):   #层层遍历
        sampleFilesDir = fileDir + '/' + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        for j in range(len(sampleList)):
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir).readlines():
                word = line.strip('\n')
                wordMap[word] = wordMap.get(word,0.0) + 1.0 #(key,value)的结构表示每个单词出现的次数
    for key, value in wordMap.items():
        if value > 3:     #只返回出现次数大于3的单词
            newWordMap[key] = value
    sortedNewWordMap = sorted(newWordMap.iteritems())
    print ('wordMap size : %d' % len(wordMap))
    print ('newWordMap size : %d' % len(sortedNewWordMap))
    return sortedNewWordMap
############################################################
def printWordMap():
    print ('Print Word Map')
    countLine=0
    fr = open('D:\\datamining\\allDicWordCountMap.txt','w')
    sortedWordMap = countWords()
    for item in sortedWordMap:
        fr.write('%s %.1f\n' % (item[0],item[1]))
        countLine += 1
    print ('sortedWordMap size : %d' % countLine)