# coding=UTF-8
import xlrd
import numpy as np
import cPickle
import sys
from collections import defaultdict

def strip_New(x):
    x = x.strip('<')	#去掉行首的<符号
    x = x.strip('>')	#去掉行尾的>符号
    x = x.split(',')	#将字符串按照','进行分割，得到entity，relation，object
    #x[0] = x[0].replace('_',' ')	#不知道是否应该处理这块
    #x[2] = x[2].replace('_',' ')
    x[1] = x[1].strip()		#去掉relation行首和行尾的' '
    x[1] = x[1].split(' ')	#将relation中的单词按照' '进行分割
    x[2] = x[2].strip()		#在object中,词首处有一个空格，需要去除
	#对relation中单词个数的判断在转换词向量的时候进行
    return x

def Read_xlsx(name = 'Bollywood_Actors.xlsx', sheet='Sheet1'):
    exfile = xlrd.open_workbook(name)

    sheet1 = exfile.sheet_by_name(sheet)  #只读取sheet1
    labels = sheet1.col_values(0)	#读取第一列
    cols_2 = sheet1.col_values(1)	#读取第二列

    #print "cols_1:"
    #print cols_1

    #print "cols_2:"
    #print cols_2

    labels = labels[1:]	#去除第一行标题
    cols_2 = cols_2[1:]	#去除第一行标题

    cols_2 = map(strip_New,cols_2) 	#利用map函数，对每个行数据进行处理
    #print cols_2

    '''
    for sentense in cols_2:
	print sentense
    '''

    #print cols_2[len(cols_2)-1][1][1]
    return labels, cols_2 #cols_2是一个二维列表，第一个维度表示第几行，第二个维度表示第几个数据；如cols_2[0][0]表示第一行数据的entity，cols_2[0][1]表示第一行数据的relation，cols_2[0][2]表示第一行数据的object

def build_data_cv(data_file, cv=10): #data_file是知识图的文件名，cv是所需要分成的份数
    """
    载入数据，并将其分成cv份
    """
    revs = []
    vocab = defaultdict(float)     #创建一个float类型的空字典
    labels, cols_2 = Read_xlsx(data_file)

    i = 0	#设置lables的循环变量
    for line in cols_2:			#在cols_2中循环，得到所有单词的集合
	vocab[line[0]] += 1
	vocab[line[2]] += 1
	for line_1 in line[1]:
	    vocab[line_1] += 1
	datum  = {  "y":labels[i], 
		    "Triple": line,
		    "split_cv": np.random.randint(0,cv),	#这两个随机数是将整个数据集分成三份，train、cv、test
			"split_test":np.random.randint(0,cv)}	#生成0-9的十个随机数
	revs.append(datum)
	i = i + 1	#每次循环之后，i自加
		
    return revs, vocab #vocab是单词字典，key是单词，value是单词的统计个数
	#revs是一个字典的列表，是对每一行的统计，其中y表示正反类别，text表示本行的所有单词，num_words表示本行的单词的个数，split是一个0-10的随机数

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}   #创建空的词向量字典
    with open(fname, "rb") as f:
        header = f.readline()  #从w2v_file文件读取整行，包括'\n'，可以指定读取多少字节数
        vocab_size, layer1_size = map(int, header.split()) #map是python内置的高阶函数，它接受一个函数f和一个list，并通过把函数f一次作用在list的每个元素上，得到一个新的list返回
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size): #xrange与range完全一样，所不同的是生成的不是一个数组，而是一个生成器;在整个w2v_file中循环
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
    return word_vecs #返回一个字典，字典的key是单词，values是对应的单词的词向量

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:  #对未知的词向量的单词，如果它出现的次数>=min_df，那么给它赋予一个值
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  #uniform是从一个均匀分布中随机采样;该均匀分布左闭右开；[-0.25,0.25)

# def get_W(word_vecs, k=300):
    # """
    # Get word matrix. W[i] is the vector for word indexed by i
    # """
    # vocab_size = len(word_vecs)
    # word_idx_map = dict()
    # W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    # W[0] = np.zeros(k, dtype='float32')
    # i = 1
    # for word in word_vecs:
        # W[i] = word_vecs[word]
        # word_idx_map[word] = i
        # i += 1
    # return W, word_idx_map
	
# revs, words = build_data_cv('Bollywood_Actors.xlsx')

# print revs[0]

# print "------------------------------------------------------------------------------"
#print words

if __name__=="__main__":    
    w2v_file = sys.argv[1]     #第一个参数传入的是已经训练好的词向量的文件位置
    data_file = sys.argv[2]    #第二个参数传入的是Knowledge Graphs文件位置

    #开始载入数据
    print "loading data...",        
    revs, vocab = build_data_cv(data_file, cv=10)

    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))  #输出单词的个数

    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)  #直接从w2v_file中读取单词的词向量，并返回，单词是从vocab中得到
	
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))  #输出vocab中部分单词的词向量（部分是因为有的单词可能在w2v_file中没有词向量）
	
    add_unknown_words(w2v, vocab) #对未知的词向量的单词，进行处理，给它赋值
    
    #print w2v['Aamir_Khan']
    #W, word_idx_map = get_W(w2v) #将词向量组成一个词矩阵，W是词矩阵，word_idx_map是单词和index的映射图
	
    # rand_vecs = {}
    # add_unknown_words(rand_vecs, vocab) #将所有的单词当做是未知单词，得到一系列的随机向量rand_vecs
    # W2, _ = get_W(rand_vecs)  #对所有的rand_vecs得到词矩阵
    cPickle.dump([w2v, revs], open("mr.p", "wb"))  #w2v是单词及其对应的词向量，revs是一个字典的列表，是对每一行的统计，其中y表示正反类别，text表示本行的所有单词，num_words表示本行的单词的个数，split是一个0-10的随机数
    print "dataset created!"
