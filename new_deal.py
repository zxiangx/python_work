import pandas as pd
import numpy as np
import torch
import torchtext as tt

#global变量
idToNum = {}#从id变为一个有序的序列
NumToid = []#从具体的编号得到其id
cb_table = 0#cb_table为最终用于训练的属性、评分,最终写入cb_table.csv文件中
kind = 0
def connect_id_Num():#创建idToNum和NumToid
    global kind
    movies = pd.read_csv("movies.csv")
    temp = np.array(movies["movieId"])
    for i in range(len(temp)):
        idToNum[temp[i]] = i
        NumToid.append(temp[i])
    kind = list(movies["genres"])#把全局变量kind存储下来，用于之后填充cb_table[:,3-22]

def build_cb_table():#初步搭建cb_table
    global cb_table
    ratings = np.array(pd.read_csv("ratings.csv"))
    cb_table = np.zeros((len(ratings),74),dtype = float)#返回结果

    ans_table = np.array(pd.read_csv("ans_table.csv"))#ans_table为爬取内容
    for i in range(len(cb_table)):
        cb_table[i,1] = idToNum[ratings[i,1]]#cb_table[:,1]为修正后的id
        cb_table[i,2] = ans_table[idToNum[ratings[i,1]],4]#cb_table[:,2]为爬取的metascore分数
    cb_table[:,0] = ratings[:,0]#cb_table[:,0]为用户id
    cb_table[:,73] = ratings[:,2]#cb_table[:,23]为评分

def fill_cb_table():#把cb_table中空余的部分补充完整
    S = set()#用于存放电影类型（str）的集合
    for i in kind:#把所有电影类型拿出来，一共20部电影
        x = i.split("|")
        for j in x:
            S.add(j)
    NumOfTag = {}#每一个Tag对应cb_table中的第几列
    accum = 2#从2开始计数，3-22列为需要填充的部位
    for i in S:
        accum += 1
        NumOfTag[i] = accum
    for i in range(len(kind)):
        Id = NumToid[i]
        temp_index = np.where(cb_table[:,1] == Id)[0]
        x = kind[i].split("|")
        for j in x:
            cb_table[temp_index, NumOfTag[j]] = 1

def deal_metascore():#处理爬取的metascore，进行填补缺失以及正则化处理
    def normalize(x,st,d):
        return (x-d)/st
    s1 = sum(cb_table[:,2] != 0)
    av1 = sum(cb_table[:,2]) / s1
    cb_table[cb_table[:,2] == 0, 2] = av1#处理缺失项
    st = np.std(cb_table[:,2])
    d = np.mean(cb_table[:,2])
    for i in range(len(cb_table)):
        cb_table[i,2] = normalize(cb_table[i,2], st, d)

def deal_tags():
    class Embedding(torch.nn.Module):#创建一个我的Embedding网络
        def __init__(self,vocab):
            super(Embedding,self).__init__()
            glove = tt.vocab.GloVe(name='6B',dim=50)#实例化glove模型
            self.embedding = torch.nn.Embedding.from_pretrained(
                glove.get_vecs_by_tokens(vocab.get_itos()),#将词表和glove模型整合
                freeze = True
            )
        def forward(self, a):
            return self.embedding(a)
    def first_deal(x):#字符串预处理，将非字母非数字的内容全部转化为空格
        x = x.lower()
        for i in range(len(x)):
            if not (('0'<=x[i] and '9'>=x[i]) or ('a'<=x[i] and 'z'>=x[i])):
                x = x[:i] + ' ' + x[i + 1:]
        return x
    tags = np.array(pd.read_csv("tags.csv"))
    tokenizer = tt.data.utils.get_tokenizer("basic_english")#初始化分词器
    word_list = [tokenizer(tags[i, 2]) for i in range(len(tags))]#分词器构建单词库
    vocab = tt.vocab.build_vocab_from_iterator(word_list)#创建词表
    E = Embedding(vocab)#初始化embedding网络
    for i in range(len(tags)):
        tags[i, 1] = idToNum[tags[i, 1]]#对于读入的tags表，先将movieid修正
        tags[i, 2] = first_deal(tags[i, 2])#处理字符串
    S = set(list(tags[:,1]))#获得所有被贴上标签的id集合
    for i in S:#i是一个电影id
        idx = np.where(tags[:,1] == i)[0]#找到tags里面id为i的索引
        length = len(idx)#length表示idx的长度，也即电影i被贴了几个标签
        temp = 0
        for j in idx:#j是索引中的一个，有tags[j,1]==i
            lt = word_list[j]#lt获取这个标签的分词列表（以空格分开）
            lt = [vocab[x] for x in lt]#将每个字转化为他的索引
            input_tensor = torch.tensor(lt)
            input_tensor = input_tensor.unsqueeze(0)
            output_tensor = E(input_tensor)
            output_tensor = output_tensor.squeeze(0)
            ttemp = 0
            for x in output_tensor:#一个标签lt可能由很多个单词组成，这里将他们
                ttemp += x
            temp += ttemp
        temp /= length
        idx = np.where(cb_table[:, 1] == i)[0]
        for j in idx:
            for k in range(50):
                cb_table[j, 23 + k] = temp[k]
        
if __name__ == "__main__":
    connect_id_Num()
    build_cb_table()
    fill_cb_table()
    deal_metascore()
    deal_tags()
    pd.DataFrame(cb_table).to_csv("cb_table2.csv")