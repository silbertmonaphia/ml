#一.毕业设计主题  
基于SSL的情感分析系统实现  
[NLP-Sentiment analysis]  
it's up to data to define a classfication criteria  

#二.数据  
京东牛奶评论.arff[tf-idf]  
谭松坡的酒店，笔记本，书语料.csv[tf-idf]

--对于监督学习分类器
80%作为训练集  
20%作为测试集

--对于半监督学习
80%作为训练集,其中的10%作为预训练数据集，90%当做无标注数据集  
20%作为测试集  
参考文献:[高伟]基于随机子空间自训练的半监督情感分类方法  

**切分训集和测试集**  
**[基于交叉验证方法]**  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)  

#三.测试
准确率，查全率，F1


#四.工具  
python2.7   
scikit,numpy,scipy  
docker[for machine learing]  

#五.算法  

##0.监督学习(SL)的分类器选择  
[能够输出后验概率的]
0.1支持向量机(SVC)  
准确率:0.8148  
0.2朴素贝叶斯－多项式分布假设(MultinomialNB)  
准确率:0.85  
0.3决策树(DecisionTreeClassifier)  
准确率:0.8241  

##1.半监督学习(SSL)  

其实还是利用回刚才那些监督学习的分类器作为基础，向外套一层逻辑  

**1.1Self-Training [完成度80%]**  
最原始的半监督学习算法，但是容易学坏,压根没有改善，甚至出现更加差
Assumption:
One's own high confidence predictions are correct.

其主要思路是首先利用小规模的标注样本训练出一个分类器，然后对未标注样本进行分类，挑选置信度(后验概率)最高的样本进行自动标注并且更新标注集，迭代式地反复训练分类器    

![Self-Training](SelfTraining.png)  

**1.2Co-Training [?]**  

##2.未来规划  

**2.1Tri-Training [?]**  
**2.2Random Subspace SSL [?]**  