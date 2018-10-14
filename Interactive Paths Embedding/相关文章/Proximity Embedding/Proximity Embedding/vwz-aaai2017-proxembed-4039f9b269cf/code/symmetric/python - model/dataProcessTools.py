#encoding=utf-8
'''
methods for processing data
'''

import numpy
import theano

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def getTrainingData(trainingDataFile):
    '''
        read training data from file
    :type string
    :param trainingDataFile
    '''
    data=[] 
    pairs=[] 
    with open(trainingDataFile) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)<=0:	#读到空行则跳过
                continue
            arr=[]
            arr.append(tmp[0]+'-'+tmp[1])	#之所以区分tmp[0]+'-'+tmp[1]与tmp[1]+'-'+tmp[0]是为了非对称关系设计的
            arr.append(tmp[1]+'-'+tmp[0])
            arr.append(tmp[0]+'-'+tmp[2])
            arr.append(tmp[2]+'-'+tmp[0])
            pairs.append(arr) 
            tmp=[int(x) for x in tmp] #将列表中的所有数据转换成int类型
            data.append(tmp)	#将tmp添加到data列表中，即是data[0]表示第一行的训练三元组，例如：4	6	3
            
    return data,pairs#pairs中全是训练数据的有序组构成

def getWordsEmbeddings(wordsEmbeddings_path):#读取通过java模块生成的特征向量x
    """
        read words embeddings from file
            a b
            c d e f ....
            g h j k ....
            a means the num(line) of the data，b means the dimension of the data
            c and g are the index of the corresponding words
            d，e，f，h，j，k，... are the content of embeddings
			a表示节点数量  b表示数据的维度 K+1+K+1
			c，g表示节点id
			d，e，f...分别表示词向量的第一个维度，第二个维度，第三个维度...
    :type String
    :param wordsEmbeddings_path
    """
    size=0
    dimension=0
    wemb=[]
    with open(wordsEmbeddings_path) as f:
        for l in f:	#按行循环
            arr=l.strip().split()	#去掉首尾的空格，并按空格进行分割
            if len(arr)==2: 	#如果为真，则表示这是文件的第一行，含有的第一个数据是图中节点的数量，第二个数据是每个节点的特征向量的维度
                size=int(arr[0])
                dimension=int(arr[1])
                wemb=numpy.zeros((size,dimension)) # @UndefinedVariable	生成一个全0的矩阵，用来存放特征向量
                continue
            id=int(arr[0])	#除了第一行外，其余行的第一个元素为节点id
            for i in range(0,dimension):	#依次将放入wemb矩阵中
                wemb[id][i]=float(arr[i+1])
    return wemb,dimension,size

def loadAllSubPaths(subpaths_file,maxlen=1000):
    """
        read all subpaths from file
    :type subpaths_file: String
    :param subpaths_file：file path 
       subpaths：1 7 1 2 3 7 其中第一个是表示开始节点id，第二个是表示结束节点id，之后就是子路径的节点的id
    :type maxlen:int
    :param maxlen:
    
    the return value is a map, and the key of this map is made of startNodeId-endNodeId.
    the value of this map is a list made of startNodeId aId bId cId dId... endNodeId
    """
    map={}
    with open(subpaths_file) as f:
        for l in f: 
            splitByTab=l.strip().split('\t')	#按照tab进行分割
            key=splitByTab[0]+'-'+splitByTab[1] 	#将开始节点id与结束节点id通过‘-’连接起来，作为返回的map的key值
            sentence=[int(y) for y in splitByTab[2].split()[:]] 	#提取子路径序列的每个节点id，将其转化为int，并放置在sentence中
            if len(sentence)>maxlen: #如果子路径的长度大于maxlen，这忽略该子路径
                continue
            if key in map:		#如果已经存在采样从节点v到节点p，那么则直接在字典的值中新增一个子路径即可
                map[key].append(sentence)	#字典的value是列表
            else: 
                tmp=[]			#如果之前不存在v-p，则行添加路径即可
                tmp.append(sentence)
                map[key]=tmp
    return map

def prepareDataForTraining(trainingDataTriples,trainingDataPairs,subpaths_map):
    """
        prepare data for training
    """
    n_triples=len(trainingDataTriples)
    
    triples_matrix=numpy.zeros([n_triples,4,2]).astype('int64')	#三维的向量，维度为n_triples,4,2
    
    maxlen=0 
    n_subpaths=0 
    allPairs=[] 
    for list in trainingDataPairs:
        for l in list:
            allPairs.append(l)
    for key in allPairs: 
        if key not in subpaths_map: 
            continue;
        list=subpaths_map[key]
        n_subpaths+=len(list) 
        for l in list:
            if len(l)>maxlen:
                maxlen=len(l)
                
    subPaths_matrix=numpy.zeros([maxlen,n_subpaths]).astype('int64') 
    
    subPaths_mask=numpy.zeros([maxlen,n_subpaths]).astype(theano.config.floatX)  # @UndefinedVariable
    
    subPaths_lens=numpy.zeros([n_subpaths,]).astype('int64')
    
    current_index=0 
    path_index=0 
    valid_triples_count=0 
    for i in range(len(trainingDataPairs)): 
        pairs=trainingDataPairs[i] 
        
        valid_triples_count+=1 
        for j in range(len(pairs)): 
            pair=pairs[j]
            list=None
            if pair in subpaths_map: 
                list=subpaths_map[pair] 
            if list is not None: 
                triples_matrix[i][j][0]=current_index
                current_index+=len(list)
                triples_matrix[i][j][1]=current_index 
                for x in range(len(list)):
                    index=path_index+x 
                    path=list[x] 
                    subPaths_lens[index]=len(path) 
                    for y in range(len(path)): 
                        subPaths_matrix[y][index]=path[y] 
                        subPaths_mask[y][index]=1. 
                path_index+=len(list) 
            else : 
                triples_matrix[i][j][0]=current_index 
                current_index+=0
                triples_matrix[i][j][1]=current_index 
                
    count=0
    for i in range(len(triples_matrix)):
        if triples_matrix[i][0][0]!=triples_matrix[i][1][1] and triples_matrix[i][2][0]!=triples_matrix[i][3][1]:
            count+=1
    triples_matrix_new=numpy.zeros([count,4,2]).astype('int64')
    index=0
    for i in range(len(triples_matrix)):
        if triples_matrix[i][0][0]!=triples_matrix[i][1][1] and triples_matrix[i][2][0]!=triples_matrix[i][3][1]:
            triples_matrix_new[index]=triples_matrix[i]
            index+=1
    triples_matrix=triples_matrix_new
    
    return triples_matrix, subPaths_matrix, subPaths_mask, subPaths_lens
    
    
def prepareDataForTest(query,candidate,subpaths_map):
    """
   prepare data for test
    """
    key1=bytes(query)+'-'+bytes(candidate)
    key2=bytes(candidate)+'-'+bytes(query)
    if key1 not in subpaths_map and key2 not in subpaths_map:
        return None,None,None
    subpaths=[]
    if key1 in subpaths_map:
        subpaths.extend(subpaths_map[key1]) 
    if key2 in subpaths_map:
        subpaths.extend(subpaths_map[key2]) 
    maxlen=0
    for subpath in subpaths:
        if len(subpath)>maxlen:
            maxlen=len(subpath)
    subPaths_matrix=numpy.zeros([maxlen,len(subpaths)]).astype('int64')
    subPaths_mask=numpy.zeros([maxlen,len(subpaths)]).astype(theano.config.floatX)  # @UndefinedVariable
    subPaths_lens=numpy.zeros([len(subpaths),]).astype('int64')
    for i in range(len(subpaths)):
        subpath=subpaths[i]
        subPaths_lens[i]=len(subpath) 
        for j in range(len(subpath)):
            subPaths_matrix[j][i]=subpath[j]
            subPaths_mask[j][i]=1.  
    
    return subPaths_matrix,subPaths_mask,subPaths_lens

    
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = numpy.arange(n, dtype="int32")	#生成从0，n-1的索引列表

    if shuffle:	#如果要打乱数据集，则打乱索引列表
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):	# //表示整数除法，返回不大于结果的最大的整数
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size	#对list进行分块，块大小为minibatch_size

    if (minibatch_start != n):	#如果最后还有剩下训练数据，则将最后剩下的训练数据当做一块
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)# 将(index,minibatch)设置成tuple（所有的tuple组成一个list），然后返回这个list
#zip将对象中对应的元素打包成一个个元组，然后返回元组的列表



