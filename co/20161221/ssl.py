#!/usr/bin/python
# coding=utf-8
from numpy import *
import multiprocessing
import codecs 
import random
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
#迭代过程
def diedai(temp):
    temp2_p0=[]#这个主要是为了最大置信度的集成学习方式做铺垫的
    temp2_p1=[]
    for ii in range(m):
        rands=array([random.randint(0, (len(myVocabList)-1)) for __ in range(m1)])
        train=array(trainMat)[:,rands]
        temp1=array(temp)[rands]
        #print temp1
        p0V,p1V,pAb = trainNB0(array(train),array(listClasses))
        p0,p1=classifyNB(array(temp1),p0V,p1V,pAb)
        temp2_p0.append(p0)
        temp2_p1.append(p1)
    p0max=max(temp2_p0)#选出置信度最高的p0样本
    p1max=max(temp2_p1)#选出置信度最高的p1样本
    if  p0max>p1max:
              #print '判断类型'
        return 0,p0max,temp
    else:
        return 1,p1max,temp
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
pos.close()
neg.close()
#测试集,训练集和未标注样本的划分
listOPosts=(text[0:100]+text[2646:2746])#训练样本，10%
listClasses=(label[0:100]+label[2646:2746])
testEntry=(text[100:300]+text[2746:2946])#测试样本，20%
testEntry_label=(label[100:300]+label[2746:2946])
unsplit=(text[300:1000]+text[2946:(2946+700)])#未标注样本 ，70%  
unsplit_label=(label[300:1000]+label[2946:(2946+700)])

#词表生成
myVocabList = createVocabList(listOPosts+testEntry+unsplit)
#生成文本向量，布尔权重
trainMat=[]#训练样本的文本向量
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
unsplitMat=[]#未标记样本的文本向量
for unsplit_line in unsplit:
    unsplitMat.append(setOfWords2Vec(myVocabList, unsplit_line)) 
testMat=[]#测试样本的文本向量
for temp in testEntry:
    testMat.append(setOfWords2Vec(myVocabList, temp))

#训练参数
m=4#特征空间的划分数目
n=4#选取前n个置信度最高的样本
m1=(len(myVocabList)/m)+1#每个空间的特征数
pool = multiprocessing.Pool(processes=14)#线程池
#a = [random.randint(0, len(myVocabList)) for __ in range(2)] 
#########半监督训练##################
#特征空间的划分
while len(unsplitMat) !=0:
      temp3_p0=[]##储存最大置信度的集成学习方式得到的p0标签,格式是numpy矩阵
      temp3_p1=[]##储存最大置信度的集成学习方式得到的p1标签,格式是numpy矩阵
      temp4_p0=[]#储存最大置信度的集成学习方式得到的p0置信度,格式是numpy矩阵
      temp4_p1=[]#储存最大置信度的集成学习方式得到的p1置信度,格式是numpy矩阵
      temp5_p0=[]#储存最大置信度的集成学习方式得到的p0向量,格式是numpy矩阵
      temp5_p1=[]#储存最大置信度的集成学习方式得到的p1向量,格式是numpy矩阵
      #迭代，并行计算
      result=pool.map(diedai, unsplitMat)#结果格式是第一个标签，第二个置信度，第三个向量
      #print result
      for ii,jj,kk in result:
          if ii==0:
              temp3_p0.append(ii)
              temp4_p0.append(jj)
              temp5_p0.append(kk)
          if ii==1:
              temp3_p1.append(ii)
              temp4_p1.append(jj)
              temp5_p1.append(kk)
      temp4_p0_index=argsort(-array(temp4_p0))#对置信度进行降序排序
      temp4_p1_index=argsort(-array(temp4_p1))
      #提取出n个正类样本和n个负类样本
      temp6_p0=array([])
      temp7_p0=array([])
      temp6_p1=array([])
      temp7_p1=array([])
      try:
         if n<len(unsplitMat):
             if n<len(temp5_p0):#取出前n个负类样本以及标签
               temp6_p0=array(temp5_p0)[temp4_p0_index[0:n],:]#储存负类向量
               temp7_p0=array(temp3_p0)[temp4_p0_index[0:n]]#储存负类标签
             else:
               temp6_p0=array(temp5_p0)#储存负类向量
               temp7_p0=array(temp3_p0)#储存负类标签
             if n<len(temp5_p1):#取出前n个正类样本以及标签
               temp6_p1=array(temp5_p1)[temp4_p1_index[0:n],:]#储存正类向量
               temp7_p1=array(temp3_p1)[temp4_p1_index[0:n]]#储存正类标签
             else:
               temp6_p1=array(temp5_p1)#储存正类向量
               temp7_p1=array(temp3_p1)#储存正类标签
             if (len(temp6_p0)!=0) and (len(temp6_p1)!=0):
               temp6=vstack((temp6_p0,temp6_p1))#提取出来的最终向量
               temp7=list(temp7_p0)+list(temp7_p1)#提取出来的最终标签
             elif (len(temp6_p0)==0) and (len(temp6_p1)!=0): 
               temp6=temp6_p1#提取出来的最终向量
               temp7=list(temp7_p1)#提取出来的最终标签
             elif (len(temp6_p0)!=0) and (len(temp6_p1)==0):
               temp6=temp6_p0#提取出来的最终向量
               temp7=list(temp7_p0)#提取出来的最终标签
             else:
               print '1'
               break
         else:
            if len(temp5_p0)!=0:
               temp6_p0=array(temp5_p0)#储存负类向量
               temp7_p0=array(temp3_p0)#储存负类标签
            if len(temp5_p1)!=0:
               temp6_p1=array(temp5_p1)#储存正类向量
               temp7_p1=array(temp3_p1)#储存正类标签
            if (len(temp6_p0)!=0) and (len(temp6_p1)!=0):
               temp6=vstack((temp6_p0,temp6_p1))#提取出来的最终向量
               temp7=list(temp7_p0)+list(temp7_p1)#提取出来的最终标签
            elif (len(temp6_p0)==0) and (len(temp6_p1)!=0): 
               temp6=temp6_p1#提取出来的最终向量
               temp7=list(temp7_p1)#提取出来的最终标签
            elif (len(temp6_p0)!=0) and (len(temp6_p1)==0):
               temp6=temp6_p0#提取出来的最终向量
               temp7=list(temp7_p0)#提取出来的最终标签
            else:
               print '0'
               break
      except:
            print len(temp6_p1),'异常',len(temp6_p0)
            break
      #将未标注样本对应的元素删除
      #for dd in temp6:
          #print dd
      #print len(temp3_p0)
      #print len(temp4_p0)
      #print len(temp5_p0)
      #print len(temp3_p1)
      #print len(temp4_p1)
      #print len(temp5_p1)
      for temp9 in temp6:
          for temp99 in unsplitMat:
              if not (temp9-temp99).any(): #相同的话(temp6-temp7).any()返回false
                  unsplitMat.remove(temp99)
      #将temp6最终向量转为列表
      temp66=[]
      for temp8 in temp6:
          temp66.append(list(temp8))
      #更新训练集和训练集标签
      trainMat+=temp66
      listClasses+=temp7
      print 'trainMat= ',len(trainMat),' unsplitMat= ',len(unsplitMat),' listClasses= ',len(listClasses)
#******************测试*********************
thisDocs=array(testMat)
test_label=[]
i=0
problity=[]#储存后验概率
p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
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
f=open('result1.txt','w')
f.write(str(sum1/len(test_label)))
f.close()
#print (len(myVocabList)/m)



