
from numpy import *
import codecs 
#创建一个带有所有单词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)
    
def setOfWords2Vec(vocabList, inputSet):
    retVocabList = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            retVocabList[vocabList.index(word)] = 1
        else:
            print 'word ',word ,'not in dict'
    return retVocabList

#另一种模型    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix,trainCatergory):
    numTrainDoc = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCatergory)/float(numTrainDoc)
    #防止多个概率的成绩当中的一个为0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDoc):
        if trainCatergory[i] == 1:
            p1Num +=trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num +=trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)#处于精度的考虑，否则很可能到限归零
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive
    
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    return p0,p1
    #if p1 > p0:
        #return 1
    #else: 
        #return 0

#************主程序***************************************************************
#导入文本
pos=codecs.open('pos.txt','r',encoding='gbk')
neg=codecs.open('neg.txt','r',encoding='gbk')
text=[]#储存文本
label=[]#储存标签
for line in pos:
    line_words=[]
    label.append(1)
    words=line.split(' ')
    for word in words:
        line_words.append(word.strip())
    text.append(line_words) 
for line in neg:
    label.append(0)
    line_words=[]
    words=line.split(' ')
    for word in words:
        line_words.append(word.strip())
    text.append(line_words)   
#测试集,训练集和未标注样本的划分
listOPosts=(text[0:100]+text[2646:2746])#训练样本，10%
listClasses=(label[0:100]+label[2646:2746])
testEntry=(text[100:300]+text[2746:2946])#测试样本，20%
testEntry_label=(label[100:300]+label[2746:2946])
unsplit=(text[300:1000]+text[2946:(2946+700)])#未标注样本 ，70%  
unsplit_label=(label[300:1000]+label[2946:(2946+700)])

#listOPosts=listOPosts1+unsplit
#listClasses=listClasses1+unsplit_label

#词表生成
myVocabList = createVocabList(listOPosts+testEntry+unsplit)
#生成文本向量
trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))#主要是生成文本向量，布尔权重
#训练
p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
#测试文本向量化
testMat=[]
for temp in testEntry:
    testMat.append(setOfWords2Vec(myVocabList, temp))
thisDocs=array(testMat)
#测试
test_label=[]
i=0
problity=[]#储存后验概率
for thisDoc in thisDocs:
    line_pro=[]
    p0,p1=classifyNB(thisDoc,p0V,p1V,pAb)
    line_pro.append(p0)
    line_pro.append(p1)
    if p0>p1:
       test_label.append(0)
       print 'classified as: ',0,p0,p1
       print '************************************'
       i+=1
    else:
       test_label.append(1)
       print 'classified as: ',1,p0,p1 
       print '************************************'
       i+=1
    problity.append(line_pro)   
#计算准确率
k=0
sum1=0.0
for kk in test_label:
    if kk==testEntry_label[k]:
       sum1+=1
       k=k+1
    else:
       k=k+1
print '准确率= ',(sum1/len(test_label))*100,'%'



