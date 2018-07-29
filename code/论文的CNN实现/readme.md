#这个目录是对The Unusual Suspects Deep Learning Based Mining of Interesting Entity Trivia from Knowledge Graphs论文的CNN方法的实现
#因为Wikipedia Dump的数据量太大，没有办法使用wiki2vec进行训练
#所以目前是使用的是已经训练好的向量:GoogleNews-vectors-negative300.bin
#本程序的运行方法如下:

#repository的大小要小于100M，所以GoogleNews-vectors-negative300.bin放不到仓库里可以到如下地址进行下载，下载完成后即可进行如下步骤
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing

#首先是从GoogleNews-vectors-negative300中读取Bollywood_Actors或者Music_Artists中所用到的词的词向量，并将其存入mr.p文件中。
python read_xlsx.py GoogleNews-vectors-negative300.bin Bollywood_Actors.xlsx
#或者是运行：
python read_xlsx.py GoogleNews-vectors-negative300.bin Music_Artists.xlsx

#之后运行train_cnn.py即可：
python train_cnn.py