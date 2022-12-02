import numpy as np
import pandas as pd
from itertools import product
from functools import reduce


#首先把数据划分成K-mer形式
def Kmers_funct(seq,k):
    X = [None]*len(seq)    #若数据只有一个序列，可不用此定义
    for i in range(len(seq)):  #若数据只有一个序列，可不用此循环
        a = seq[i]
        t=0
        l=[]
        for index in range(len(a)):
            t=a[index:index+k]
            if (len(t))==k:
                l.append(t)
        X[i] = l
    return np.array(X)  #具体看返回需要，也可直接：return X

#提取核苷酸类型（排列组合）
def nucleotide_type(k):
    z = []
    for i in product('ACGU', repeat = k):  #笛卡尔积（有放回抽样排列）
        z.append(''.join(i))  #把('A,A,A')转变成（AAA）形式
    return z

#定义K-mer频率模块
def Kmers_frequency(seq,x):
    X = []
    char = reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'U', 'C', 'G']] * x)  #调用提取核苷酸类型（排列组合）代码
    #char = nucleotide_type(x)   #调用提取核苷酸类型（排列组合）代码
    for i in range(len(seq)):
        s = seq[i]
        frequence = []
        for a in char:
            number = s.count(a)  #依次统计字符数量
            #sum = L-K+1
            #kmersum = len(seq) - 3 + 1
            char_frequence = number/len(s)  #计算频率
            frequence.append(char_frequence)
        X.append(frequence)
    return X
#定义K-mer在总体中所占比例频率模块
def Kmers_sumfrequency(seq,x,sum2):
    X = []
    sum3 = 0
    sum33 = 0
    char = reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'U', 'C', 'G']] * x)  # 调用提取核苷酸类型（排列组合）代码
    # char = nucleotide_type(x)   #调用提取核苷酸类型（排列组合）代码

    for a in char:
        for i in range(len(seq)):
            s = seq[i]
            number = s.count(a)  # 依次统计每行字符数量，例如：AAA在第一行的个数
            # sum = L-K+1
            # kmersum = len(seq) - 3 + 1
            sum3 = sum3+number
        print(a,sum3) #sum3出来之后就是AAA在所有数据中有的个数
        char_frequence = sum3 / sum2  # 计算AAA在总体中所占的比例
        X.append(char_frequence)
    return X


if __name__ == '__main__':
    k = 3
    miRNA = pd.read_csv(r'miRBaseSequence.csv')
    seq = miRNA['sequence']
    print(len(seq)) #输出有1152行
    seq_kmer=Kmers_funct(seq,k)
    print(len(seq_kmer))#输出有1152行
    #统计miRNA所有的k-mers数
    resoult = {}
    sum1 = 0
    sum2 = 0
    j=0
    for i in range(len(seq_kmer)):
        s = seq_kmer[i]
        j = j+1
        for i in s:
            resoult[i] = s.count(i)
        #print(resoult)
        # 把键值对中的值拿出来，单独相加就求出来了
        values = resoult.values()
        for value in values:
            sum1 = sum1 + value
        #print(j,sum1)
        sum2 = sum2+sum1
    print("sum2",sum2)

    feature_kmer=Kmers_frequency(seq_kmer,k) #转换为频率向量
    Kmers_sumf = Kmers_sumfrequency(seq_kmer,k,sum2)#每个k-mer对应在总体k-mers数量中占据的比例

    #print(feature_kmer,len(feature_kmer))
    pd.DataFrame(feature_kmer).to_csv(r'miRNA_fvector.csv', header=None, index=None)
    pd.DataFrame(seq_kmer).to_csv(r'miRNAseq_Kmers.csv', header=None, index=None)
    pd.DataFrame(Kmers_sumf).to_csv(r'miRNAsum_Kmerf.csv', header=None, index=None)

