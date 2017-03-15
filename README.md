## 基于贝叶斯算法的中文情感分类
- 工具：PyDev+Anaconda3+jieba
- 准确率：3000条语料2309条预测正确，约为77%
- 所使用的语料来自github
### 代码说明
1. 导入语料与分词等处理与基于词典的中文情感分类相同</br>
[基于词典的中文情感分类](https://github.com/panhaiqi/textpredition)
2. 编写贝叶斯算法类，并创建默认构造方法
<pre><code>
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
</pre></code>
2.导入和训练数据集，生成算法必须的参数和数据结构
<pre>
    def train_set(self,trainset,classvec):
        self.cate_prob(classvec)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]
        self.vocabulary = list(tempset)
        self.vocablen = len(self.vocabulary)
        self.calc_wordfreq(trainset)
        self.build_tdm()
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
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx,self.vocabulary.index(word)] += 1
            for singleworld in set(trainset[indx]):
                self.idf[0,self.vocabulary.index(singleworld)] += 1
</pre>
5.按分类计算向量空间的每维值P(x|yi)
<pre>
    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates),self.vocablen])
        sumlist = np.zeros([len(self.Pcates),1])
        for indx in range(self.doclength):
            self.tdm[self.labels[indx]] += self.tf[indx]
            sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])
        self.tdm = self.tdm/sumlist
</pre>
6.将测试集映射到当前词典
<pre>
   def map2vocab(self,testdata):
        self.testset = np.zeros([1,self.vocablen])
        for word in testdata:
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
nb.train_set(splitdata,listclass)
for i in range(len(splitdata)):
    nb.map2vocab(splitdata[i])
    if nb.predict(nb.testset) == listclass[i]:
        count += 1
print ('rate:',count/len(splitdata))
</pre>

