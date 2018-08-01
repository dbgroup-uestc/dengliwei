#coding=UTF-8
import keras
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Input, Dense, Flatten, Dropout, concatenate
import numpy as np
import cPickle
import random
from collections import defaultdict
from math import log
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

import sys

reload(sys)

sys.setdefaultencoding('utf8')

def triple2vec(Atriple, w2v): #传入一个三元组，将该三元组<e,p,o>转换成向量，其中p中的单词数为8，如果不够则用0向量补齐，如果多了，则截断
    vec = []
    vec.append(w2v[Atriple[0]].astype(float))
    vec.append(w2v[Atriple[2]].astype(float))
    if len(Atriple[1]) >= 8:
        i = 0
        for word in Atriple[1]:
            if i < 8:
                vec.append(w2v[word].astype(float))
                i = i + 1
    elif len(Atriple[1]) < 8 and len(Atriple[1]) >= 1:
        i = 0
        for word in Atriple[1]:
			vec.append(w2v[word].astype(float))
			i = i + 1
		#生成一个300维的0向量
        zeros = [0 for j in range(300)]
        zeros = np.array(zeros, dtype=float)
        while i < 8:
            vec.append(zeros)
            i = i + 1
    return vec

def splitdata(revs, cv, w2v):
    """
    Transforms sentences into a 2-d matrix.
    """
    x_train, x_test, x_cv = [], [], []		#训练集、测试集、cv集的数据
    y_train, y_test, y_cv = [], [], []	#训练集、测试集、cv集的标签
    for rev in revs:  #对所有的text进行操作
        if rev["split_cv"]==cv:            #如果正好split==cv，那么就将三元组作为test集
            x_cv.append(triple2vec(rev['Triple'],w2v))
            y_cv.append(rev['y'])
        elif rev["split_test"]==cv:
			x_test.append(triple2vec(rev['Triple'],w2v))
			y_test.append(rev['y'])
        else:  
            x_train.append(triple2vec(rev['Triple'],w2v))     #否则，就将三元组作为训练集
            y_train.append(rev['y'])
    return x_train, x_test, y_train, y_test, x_cv, y_cv    #返回训练集与测试集组成的列表

#该函数用于计算IEF
def IEF_Func(predicate_num, D):
    return_list = []
    for predicate in predicate_num:
        return_list.append(log(float(D)/predicate))
    return return_list

#该函数用于计算PF
def PF_Func(predicate_dict, entity_dict, revs):
    PF_list = []
    for entity in entity_dict:
        for predicate in predicate_dict:
            i = 0
            for rev in revs:
                if rev['Triple'][0] == entity and predicate == " ".join(rev['Triple'][1]):
                    i += 1
            PF_list.append(i)
    return PF_list

#输入三元组与词袋，将三元组的predicate转化成向量列表
def bow2vec(triple, bag_of_words):
    x = [0 for i in range(len(bag_of_words))]
    for predicate in triple[1]:
        i = 0
        for key in bag_of_words.keys():
            if key == predicate:
                x[i] += 1
            i += 1
    return x

def get_index(obj, dict):
    i = 0
    for d in dict:
        if obj == d:
            return i
        i += 1

#从样本计算出x，以及y
def CaculateXY(samples, bag_of_words):
    x = [[] for i in range(len(samples))]  
    y = []
    inlinks_entity, inlink_entity_dict = cPickle.load(open("inlinks_e.p","rb"))
    inlinks_object, inlink_object_dict = cPickle.load(open("inlinks_o.p","rb"))

    # 得到entity、predicate、object的字典；其value就是该word出现的次数
    entity_dict = defaultdict(int)
    predicate_dict = defaultdict(int)
    object_dict = defaultdict(int)

    for rev in samples:
        entity_dict[rev['Triple'][0]] += 1
        predicate_dict[" ".join(rev['Triple'][1])] += 1
        object_dict[rev['Triple'][2]] += 1
        y.append(rev['y'])

    D = len(entity_dict)
    # 计算IEF
    IEF_list = IEF_Func(predicate_dict.values(), D)  # 这是一个一维列表，长度为len(predicate_dict)
    IEF = np.array(IEF_list)
    # 计算PF
    PF_list = PF_Func(predicate_dict, entity_dict, samples)  # 这是个一维列表，大小为len(entity_dict)*len(predicate_dict)
    PF = np.array(PF_list).reshape(len(entity_dict), len(predicate_dict))
    PF_IEF = []
    for i in range(len(entity_dict)):
        PF_IEF.append(PF[i] * IEF)

    #    PF_IEF = np.array(PF_IEF)
    #    IEF = IEF.tolist()
    #    PF_IEF = PF_IEF.tolist()

    i = 0
    for rev in samples:
        j = get_index(rev['Triple'][0], entity_dict)
        k = get_index(" ".join(rev['Triple'][1]), predicate_dict)
        x[i].append(IEF[k])
        x[i].append(PF_IEF[j][k])
        i += 1

    # 计算wikipedia上的点击次数
    i = 0
    for rev in samples:
        x[i].append(inlinks_entity[get_index(rev['Triple'][0], inlink_entity_dict)])
        x[i].append(inlinks_object[get_index(rev['Triple'][2], inlink_object_dict)])
        i += 1

    #从词袋中生成向量
    i = 0
    for rev in samples:
        for t in bow2vec(rev['Triple'], bag_of_words):
            x[i].append(float(t))
        i += 1

    return x, y
    
def splitdata_HCF(revs, cv, HFC_X):
    x_train, x_test, x_cv = [], [], []		#训练集、测试集、cv集的数据
    i = 0
    for rev in revs:  #对所有的text进行操作
        if rev["split_cv"]==cv:            #如果正好split==cv，那么就将三元组作为test集
            x_cv.append(HFC_X[i])
        elif rev["split_test"]==cv:
			x_test.append(HFC_X[i])
        else:  
            x_train.append(HFC_X[i])     #否则，就将三元组作为训练集
        i += 1
    return np.array(x_train), np.array(x_test), np.array(x_cv)  #返回训练集与测试集组成的列表
    
if __name__=="__main__":
    print "loading data..."
    x = cPickle.load(open("mr.p","rb"))		#加载之前处理好的数据
    w2v, revs = x[0], x[1]					#数据还原\
    print "data loaded!"
    
    #获取 HCF 
    samples = random.sample(revs, 5000)    #获得5000个随机样本
    bag_of_words = defaultdict(int)
    for rev in samples:
        for predicate in rev['Triple'][1]:
            bag_of_words[predicate] += 1
#    samples = revs

    x_sample, y_sample = CaculateXY(samples, bag_of_words)

    #利用SVM-L选取特征
    lsvc = LinearSVC(C=0.4, penalty='l1', dual=False).fit(x_sample, y_sample)
    model = SelectFromModel(lsvc, prefit=True)
    
    #x_new = model.transform(x_sample)

    #对于每个输入的<e,p,o>算出各自的 HCF_X
    X, Y = CaculateXY(revs, bag_of_words)
    HCF_X = model.transform(X)
    HCF_X = HFC_X[:,0:100]  #选取前100维的特征向量
    
    #设置训练参数
    #batch_size = 128    
    batch_size = 256
    epochs = 12   #设置训练次数
    pool_size = 2
    kernel_size = 3
    strides = 1
    
    results = []
    
    # dimension of input data
    img_rows, img_cols = 10, 300   #(8+2)*300
    input_shape = (img_rows, img_cols)
    input_shape2 = (1000)
    
    for cv in range(10):		#将数据组成10种组合，然后训练
        x_train, x_test, y_train, y_test, x_cv, y_cv = splitdata(revs, cv, w2v)  #将整个三元组集合分成训练集和测试集
        #x_train samplenum*10*300     y samplenum*1
        
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        x_cv = np.array(x_cv)

        print 'x_train shape:', x_train.shape
        print x_train.shape[0], 'train samples'
        print x_cv.shape[0], 'cv samples'
        print x_test.shape[0], 'test samples'
        
        #将HCF同样的进行划分
        HCF_train, HCF_test, HCF_cv = splitdata_HCF(revs, cv, HCF_X)

        #conv+Max_pool+conv+Max_pool+Flatten+Merge+Full_Connected*4
        #首先定义输入层
        input = Input(shape = input_shape)
        x = Conv1D(filters=300,kernel_size=kernel_size,activation='relu')(input)
        x = MaxPooling1D(pool_size=pool_size,strides=None)(x)
        x = Dropout(0.1)(x) #设置 Dropout 率为0.1
        x = Conv1D(filters=300,kernel_size=kernel_size,activation='relu')(x)
        x = MaxPooling1D(pool_size=pool_size,strides=None)(x)
        x = Dropout(0.1)(x)
        x = Flatten()(x)
        
        #增加HCF的输入层
        input_2 = Input(shape = (100,))
        x = concatenate([input_2, x], axis=1)  #链接 HCF 与从CNN中提取出的特征
        
        #四个全连接层
        x = Dense(400,activation='relu')(x)
        x = Dense(400,activation='relu')(x)
        x = Dense(400,activation='relu')(x)
        output = Dense(1,activation='relu')(x)
        
        model = Model(inputs=[input, input_2], outputs=output)
        
        sgd = keras.optimizers.SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)  #Optimizer=sgd , momentum=0.8
                 
        #模型编译，设置相应的训练参数
        model.compile(loss='binary_crossentropy',
                       optimizer=sgd,
                       metrics=['accuracy']) 
 #
        model.fit([x_train,HCF_train], y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=([x_cv,HCF_cv], y_cv))   #verbose：日志显示，0表示不在标准输出流输出日志信息，1为输出进度条记录
        score = model.evaluate([x_test,HCF_test], y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        results.append(score)
#        
#        
        
        
    
