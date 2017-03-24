## 基于贝叶斯算法的中文情感分类
- 工具：PyDev+Anaconda3+jieba
- 准确率：3000条语料随机分成训练集和测试集，准确率稳定在76%左右
- 所使用的语料来自github
### 代码说明
1. 导入语料与分词等处理与基于词典的中文情感分类相同</br>
[基于词典的中文情感分类](https://github.com/panhaiqi/textpredition)
2. 编写贝叶斯算法类，并创建默认构造方法
<pre><code>
class NBayes(object):
    def __init__(self):
        self.vocabulary = []#词典
        self.idf = 0
        self.tf = 0        #训练集的权值矩阵
        self.tdm = 0       #p(x|yi)
        self.Pcates = {}   #p(yi)类别词典
        self.labels = []   #对应每个文本的分类
        self.doclength = 0 #训练集文本数
        self.vocablen = 0  #词典词长
        self.testset = 0   #测试集
</pre></code>
2.导入和训练数据集，生成算法必须的参数和数据结构
<pre>
    def train_set(self,trainset,classvec):
        self.cate_prob(classvec)        #计算每个分类在数据集中的概率p(x|yi)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]
        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        self.calc_wordfreq(trainset)    #计算词频数据集
        self.build_tdm()                #按分类累计向量空间的每维值p(x|yi)
</pre>
3.计算数据集中每个分类的概率P(yi)
<pre>
   def cate_prob(self,classvec):
        self.labels = classvec
        labeltemps = set(self.labels)
        for labeltemp in labeltemps:
            self.Pcates[labeltemp] = float(self.labels.count(labeltemp))/float(len(self.labels))
</pre>
4.生成普通的词频向量
<pre>
   def calc_wordfreq(self,trainset):
        self.idf = np.zeros([1,self.vocablen])    
        self.tf = np.zeros([self.doclength,self.vocablen])
        for indx in range(self.doclength):#遍历所有文本
            for word in trainset[indx]:#遍历文本中的每个词
                self.tf[indx,self.vocabulary.index(word)] += 1  #找到文本的词在词典中加1
            for singleworld in set(trainset[indx]):
                self.idf[0,self.vocabulary.index(singleworld)] += 1
</pre>
5.按分类计算向量空间的每维值P(x|yi)
<pre>
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates),self.vocablen])   #类别行*词典列
        sumlist = np.zeros([len(self.Pcates),1])                #统计每个分类的总值
        for indx in range(self.doclength):                      #将同一类别的词向量空间值加总
            self.tdm[self.labels[indx]] += self.tf[indx]        #统计每个分类的总值
            sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])
        self.tdm = self.tdm/sumlist
</pre>
6.将测试集映射到当前词典
<pre>
    def map2vocab(self,testdata):
        self.testset = np.zeros([1,self.vocablen])
        for word in testdata:
            if word in self.vocabulary:
                self.testset[0,self.vocabulary.index(word)] += 1
</pre>
7.预测分类结果，输出预测的分类类别
<pre>
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
</pre>
8.获取执行结果
<pre>
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
</pre>

