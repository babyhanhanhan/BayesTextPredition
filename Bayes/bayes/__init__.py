# -*- coding: utf-8 -*-
import sys
import os
import codecs
import jieba
from numpy import *
import numpy as np

f = codecs.open("D:\\emotion-analysis\\data\\f_4.txt",'r',encoding='utf-8')
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
######################################################################################
f = codecs.open("D:\\emotion-analysis\\data\\f_1.txt",'r',encoding='utf-8')
data1 = f.readlines()
newdata1 = []
segmenttest1 = []
newsegmenttest1 = []
for line in data1:
    newdata1.append(line.strip('\r\n'))
for line in newdata1:
    segmenttest1.append(line.replace('\t',''))
    
classpositive1 = []
classnegative1 = []
classneutral1 = []

temp11 = []
temp21 = []
temp31 = []

for line in segmenttest1:
    if line[0] == '1':
        classpositive1.append(line[1:])
    if line[0] == '0':
        classneutral1.append(line[1:])
    if line[0] == '-':
        classnegative1.append(line[2:])
        
splitpositive1 = []
splitnegative1 = []
splitneutral1 = []


for line in classpositive1:
    temp11.append(jieba.cut(line))
for line in classneutral1:
    temp21.append(jieba.cut(line))
for line in classnegative1:
    temp31.append(jieba.cut(line))
    
i1 = 0
i2 = 0
i3 = 0
for line in temp11:
    splitpositive1.append([])
    for seg in line:
        if seg != '\r\n':
            splitpositive1[i1].append(seg)
    i1 += 1      
for line in temp21:
    splitneutral1.append([])
    for seg in line:
        if seg != '\r\n':
            splitneutral1[i2].append(seg)
    i2 += 1
for line in temp31:
    splitnegative1.append([])
    for seg in line:
        if seg != '\r\n':
            splitnegative1[i3].append(seg)
    i3 += 1

splitdata1 = splitpositive1 + splitnegative1 +splitneutral1
listclass1 = []
for line in splitpositive1:
    listclass1.append(1)
for line in splitnegative1:
    listclass1.append(2)
for line in splitneutral1:
    listclass1.append(0)
print(listclass1)
##############################################################################
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
            if word in self.vocabulary:
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
newlistclass = []
nb.train_set(splitdata,listclass)
for i in range(len(splitdata1)):
    nb.map2vocab(splitdata1[i])
    newlistclass.append(nb.predict(nb.testset))
    if nb.predict(nb.testset) == listclass1[i]:
        count += 1
#print (listclass)
print (newlistclass)
print ('rate:',count/len(splitdata1))