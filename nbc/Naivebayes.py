import os
import string
import numpy
import datetime
import random

from tkinter import _flatten
from textblob import TextBlob
from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

##########################################获取路径############################################

def getpath(path,rate):
    ##  返回两个字典（train|test）和训练集元组
    list=[]
    lable=0
    dict={}
    os.chdir(path)
    fd=os.listdir()
    for each in fd:
        fdpath=path+'\\'+each
        f=os.listdir(fdpath)
        for each in f:
            fpath=fdpath+'\\'+each
            list.append(fpath)
        dict.setdefault(lable,fplist)
        fplist=[]
        lable+=1

    train={}
    test={}
    l = 0
    for key in dict.keys():
       r=random.randint(1,99)
       trainlist,testlist=train_test_split(dict[key], test_size=rate, random_state=r)
       train.setdefault(key,trainlist)
       test.setdefault(key,testlist)
       l+=len(trainlist)
    traindoc=l

    return train,test,traindoc

def read(path):
 ##   读出文档并储存，返回字典值为以每篇文章为元素的列表
    list=[]
    dict={}
    for key in path.keys():
        for each in path[key]:
            f=open(each,"rb")
            fr = f.read()
            frdecode = fr.decode('ISO-8859-1')
            list.append(frdecode)
            f.close()
        dict.setdefault(key,list)
        list=[]
    return dict

 ######################################预处理#####################################
def sy(docwordlist):
    ll_wordlist = []
    for each in docwordlist:
        ll_wordlist.append(str.lower(each))
    return ll_wordlist

def tokener(doc):
    tbdoc = TextBlob(doc)
    doclist=tbdoc.words
    return doclist

def cleanlines(doc):
    intab= string.digits+string.punctuation
    outtab = " "*len(string.digits + string.punctuation)
    maketrans = str.maketrans(intab,outtab)
    cldoc = doc.translate(maketrans)
    return cldoc

def lemmatize(doclist):
    lm = []
    wnl = WordNetLemmatizer()
    for each in doclist:
        lm.append(wnl.lemmatize(each))
    return lm

def stem(docwordlist):
    stlist = []
    stemmer = SnowballStemmer("english")
    for each in docwordlist:
        stlist.append(stemmer.stem(each))
    return stlist

def stopwords(docwordlist):
    drlist = [w for w in docwordlist if w not in stopwords.words('english') and 3<len(w)]
    return drlist

def preprocess(fdict):
    dict = {}
    for key in fdict.keys():
        list = []
        for doc in fdict[key]:
            clean = cleanlines(doc)
            tokener = tokener(clean)
            lemmatize = lemmatize(tokener)
            sy = sy(lemmatize)
            steming = stem(sy)
            list = stopwords(steming)
            list.append(list)
        dict.setdefault(key,list)
    return dict

#####################################滤掉高低频词#####################################

def frequency(traindict,low,high):
    for key1 in traindict.keys():
        wordlist=list(_flatten(traindict[key1]))
        frequency = dict(Counter(wordlist))
        record=[]
        for key in frequency.keys():
            if frequency[key]<low or frequency[key]>high:
                record.append(key)
        for key in record:    
            frequency.pop(key)
        for doclist in traindict[key1]:
            for word in doclist:
                if word not in list(frequency.values()) :
                    doclist.remove(word)
    return traindict
                   
##################################分类并计算正确率##############################

def NBC(traindict,testdict,trainnum):
    i=0
    j=0
    for key2 in testdict.keys():
        testclass=[]
        count=0
        for doclist in testdict[key2]:
            pdict={}
            for key1 in traindict.keys():
                prior=numpy.log(len(traindict[key1])/trainnum)
                classlist=list(_flatten(traindict[key1]))
                frequencydict = dict(Counter(classlist))
                total=len(classlist)
                classwordnum=len(list(set(classlist)))
                
                otraindict = traindict.copy()
                otraindict.pop(key1)
                allotherlist=[]
                for key3 in otraindict.keys():
                    eachclasslist=list(_flatten(otraindict[key3]))
                    allotherlist.append(eachclasslist)
                otherclasslist=list(_flatten(allotherlist))
                othertotal=len(otherclasslist)
                frequency = dict(Counter(otherclasslist))
                num=len(list(set(otherclasslist)))
            
                eachdoc=0
                for each in doclist:
                    wordcount=frequencydict[each] if each in frequencydict.keys() else 0
                    othercount=frequency[each] if each in frequency.keys() else 0
                    eachclass=numpy.log((wordcount+1)/(total+classwordnum))
                    eachnotinclass=numpy.log((othercount+1)/(othertotal+num))
                    eachword=eachclass-eachnotinclass
                    eachdoc+=eachword
                eachdoc+=prior
                pdict.setdefault(key1,eachdoc)
            p=max(pdict,key=pdict.get)
            testclass.append(p)
        for each in testclass:
            if each==key2:
                count+=1
        n=count/len(testclass)
        i+=count
        j+=len(testclass)
    acc=i/j
    return acc

def main():
    path="D:data mining\\20news-18828"
    rate=0.2
    dict=getpath(path,rate)
    traindict=dict[0]
    testdict=dict[1]
    traindocnum=dict[2]
    trainfile=read(traindict)
    testfile=read(testdict)
    trainword=preprocess(trainfile)
    testfiledict=preprocess(testfile)
    trainworddict=frequency(trainword,2,5000)
    acc=NBC(trainworddict,testfiledict,traindocnum)
    return acc

time=1
l=0.0
for t in range(time):
    acc=main()
    l+=acc
ave=l/time
print('The NBC average result is %f'%ave)
  





   