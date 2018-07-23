import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "rb") as f:
        for line in f:   #对整个数据集按行循环    
            rev = []
            rev.append(line.strip())#strip用于移除字符串头尾指定的字符，默认是空格和换行符号
            if clean_string:
                orig_rev = clean_str(" ".join(rev))#在每一行行首加上一个空格，并调用clearn_str
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())#以空格作为分割得到单个单词
            for word in words:
                vocab[word] += 1#统计单词个数
            datum  = {"y":1, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}#生成0-9的十个随机数
            revs.append(datum)
    with open(neg_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split":  np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab #vocab是单词字典，key是单词，value是单词的统计个数
	#revs是一个字典的集合，是对每一行的统计，其中y表示正反类别，text表示本行的所有单词，num_words表示本行的单词的个数，split是一个0-10的随机数
    
def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}   #创建空的词向量字典
    with open(fname, "rb") as f:
        header = f.readline()  #从文件读取整行，包括'\n'，可以指定读取多少字节数
        vocab_size, layer1_size = map(int, header.split()) #map是python内置的高阶函数，它接受一个函数f和一个list，并通过把函数f一次作用在list的每个元素上，得到一个新的list返回
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size): #xrange与range完全一样，所不同的是生成的不是一个数组，而是一个生成器
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:  #对未知的词向量的单词，如果它出现的次数>=min_df，那么给它赋予一个值
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  #uniform是从一个均匀分布中随机采样;该均匀分布左闭右开；[-0.25,0.25)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """#将字符串进行正则化处理
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = sys.argv[1]     """word2vec bin"""
    data_folder = ["rt-polarity.pos","rt-polarity.neg"]    #pos是正面的材料 neg是负面的材料
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=True)
    max_l = np.max(pd.DataFrame(revs)["num_words"]) #按照num_words所给的数值，得到整个材料（pos+neg）中句子的最大长度
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))  #输出单词的个数
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)  #直接从w2v_file中读取单词的词向量，并返回，单词是从vocab中得到
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))  #输出vocab中部分单词的词向量（部分是因为有的单词可能在w2v_file中没有词向量）
    add_unknown_words(w2v, vocab) #对未知的词向量的单词，进行处理，给它赋值
    W, word_idx_map = get_W(w2v) #将词向量组成一个词矩阵，W是词矩阵，word_idx_map是单词和index的映射图
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab) #将所有的单词当做是未知单词，得到一系列的随机向量rand_vecs
    W2, _ = get_W(rand_vecs)  #对所有的rand_vecs得到词矩阵
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"
    #cPickle可以对任意一种类型的python对象进行序列化操作。而所谓的序列化，我们可以粗浅的理解为就是为了能够完整的保存并能够完全可逆的恢复
	#revs是一个字典的集合，是对每一行的统计。对该集合中的每一个字典而言，其中y表示正反类别，text表示本行的所有单词，num_words表示本行的单词的个数，split是一个0-10的随机数
	#W是w2v中有的单词的词向量+未知单词的随机词向量组成的矩阵
	#W2是完全由未知单词的随机词向量组成的矩阵
	#word_idx_map是单词到index的映射图
	#vocab是单词字典，key是单词，value是单词的统计个数