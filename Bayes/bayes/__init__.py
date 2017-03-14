# -*- coding: utf-8 -*-
import sys
import os
import codecs
import jieba
from numpy import *
import numpy as np

f = codecs.open("D:\\emotion-analysis\\data\\pnn_annotated.txt",'r',encoding='utf-8')
data = f.readlines()
newdata = []
segmenttest = []
newsegmenttest = []
for line in data:
    newdata.append(line.strip('\r\n'))
for line in newdata:
    segmenttest.append(line.replace('\t',''))
    
classpositive = []
classnegative = []
classneutral = []

temp1 = []
temp2 = []
temp3 = []

for line in segmenttest:
    if line[0] == '1':
        classpositive.append(line[1:])
    if line[0] == '0':
        classneutral.append(line[1:])
    if line[0] == '-':
        classnegative.append(line[2:])
        
splitpositive = []
splitnegative = []
splitneutral = []


for line in classpositive:
    temp1.append(jieba.cut(line))
for line in classneutral:
    temp2.append(jieba.cut(line))
for line in classnegative:
    temp3.append(jieba.cut(line))
    
i1 = 0
i2 = 0
i3 = 0
for line in temp1:
    splitpositive.append([])
    for seg in line:
        if seg != '\r\n':
            splitpositive[i1].append(seg)
    i1 += 1      
for line in temp2:
    splitneutral.append([])
    for seg in line:
        if seg != '\r\n':
            splitneutral[i2].append(seg)
    i2 += 1
for line in temp3:
    splitnegative.append([])
    for seg in line:
        if seg != '\r\n':
            splitnegative[i3].append(seg)
    i3 += 1

splitdata = splitpositive + splitnegative +splitneutral
listclass = []
for line in splitpositive:
    listclass.append(1)
for line in splitnegative:
    listclass.append(2)
for line in splitneutral:
    listclass.append(0)
print(splitdata)
print(listclass)
f1 = codecs.open("D:\\textpredition\\emotionanalyse\\positive.txt",'r',encoding='utf-8')
data1 = f1.readlines()
newdata1 = data1[1:]

f2 = codecs.open("D:\\textpredition\\emotionanalyse\\negative.txt",'r',encoding='utf-8')
data2 = f2.readlines()

newdata2 = data2[1:]

segment1 = []
segment2 = []
segment = []

for line in newdata1:
    segment1.append(line.strip(' \r\n'))
for line in newdata2:
    segment2.append(line.strip(' \r\n'))


class NBayes(object):
    def __init__(self):
        self.vocabulary = []
        self.idf = 0
        self.tf = 0
        self.tdm = 0
        self.Pcates = {}
        self.labels = []
        self.doclength = 0
        self.vocablen = 0
        self.testset = 0

    def train_set(self,trainset,classvec):
        self.cate_prob(classvec)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]
        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        self.calc_wordfreq(trainset)
        self.build_tdm()

    def cate_prob(self,classvec):
        self.labels = classvec
        labeltemps = set(self.labels)
        for labeltemp in labeltemps:
            self.Pcates[labeltemp] = float(self.labels.count(labeltemp))/float(len(self.labels))
        
    def calc_wordfreq(self,trainset):
        self.idf = np.zeros([1,self.vocablen])
        self.tf = np.zeros([self.doclength,self.vocablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx,self.vocabulary.index(word)] += 1
            for singleworld in set(trainset[indx]):
                self.idf[0,self.vocabulary.index(singleworld)] += 1

    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates),self.vocablen])
        sumlist = np.zeros([len(self.Pcates),1])
        for indx in range(self.doclength):
            self.tdm[self.labels[indx]] += self.tf[indx]
            sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])
        self.tdm = self.tdm/sumlist

    def map2vocab(self,testdata):
        self.testset = np.zeros([1,self.vocablen])
        for word in testdata:
            self.testset[0,self.vocabulary.index(word)] += 1
        
    def predict(self,testset):
        if np.shape(testset)[1] != self.vocablen:
            print ('输入错误')
            exit(0)
        predvalue = 0
        predclass = ''
        for tdm_vect,keyclass in zip(self.tdm,self.Pcates):
            temp = np.sum(testset*tdm_vect*self.Pcates[keyclass])
            if temp > predvalue:
                predvalue = temp
                predclass = keyclass
        return predclass
    
count = 0
nb = NBayes() 
nb.train_set(splitdata,listclass)
for i in range(len(splitdata)):
    nb.map2vocab(splitdata[i])
    if nb.predict(nb.testset) == listclass[i]:
        count += 1
print ('rate:',count/len(splitdata))