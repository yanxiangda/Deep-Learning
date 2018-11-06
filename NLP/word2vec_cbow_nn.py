

# Continues bag-of-words的神经网络实现方法

import pandas as pd, numpy as np, os
import jieba
os.chdir("D:/github_yxd/Deep-Learning/NLP")

stopwords = pd.read_table("stopword.txt", encoding='utf-8', sep='\br')
stopwords.columns = ['stopword']
stopwords = list(stopwords.stopword)
stopwords.extend([' ', '\n'])

simpleStopWords = [' ', '\n', ',', '.', ':', ';']


dataset = '''
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
'''
cuts = list(jieba.cut(dataset, cut_all=False))

returnCuts = []
for i in cuts:
    if i in simpleStopWords:
        pass
    else:
        returnCuts.append(i)
nlpData = pd.DataFrame({'word':returnCuts})

oneHot = pd.get_dummies(list(set(returnCuts)))

# 定义半个窗口大小c, 前c个和后c个单词
for i in range(nlpData.size):
    nlpData


# 学习速率
learningRate = 0.5
# 定义隐含层神经元数量
V = len(oneHot.columns)
N = 50
# 初始化权值
WI = np.random.normal(0,0.1,(V,N))
WO = np.random.normal(0,0.1,(N,V))
# 向前传播公式
# 输出的参数都是numpy形式, 且维度均为1*V的一个行向量
def cbowSGPOpt(inputVec, outputVec):
    global WI, WO
    # print (WI[0])
    # 向前传播部分
    h = np.dot(inputVec, WI)
    u = np.dot(h, WO)
    eu = np.exp(u)
    eusum = np.sum(eu)
    yPred = eu / eusum
    jOpt = list(outputVec).index(1)
    e = yPred.copy()
    e[jOpt] = e[jOpt] - 1 
    # print (e)
    # 反向传播部分
    WO = [[WO[i][j] - learningRate * h[i] * e[j] for j in range(V)] for i in range(N)]
    WI = [[WI[k][i] - learningRate * np.sum([e[j] * WO[i][j] for j in range(V)]) * inputVec[k] for i in range(N)] for k in range(V)]
    # print (WI[0])
    return yPred

# WI, WO = cbowSGPOpt(oneHot['abundance'], oneHot['accounted']

for i in range(10):
    yPred = cbowSGPOpt(oneHot['abundance'], oneHot['accounted'])
    print (np.argmax(yPred))



