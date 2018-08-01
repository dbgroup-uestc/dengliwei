本目录下的程序是对The Unusual Suspects Deep Learning Based Mining of Interesting Entity Trivia from Knowledge Graphs论文的F-CNN方法进行实现

其中对于HCF中的popularity特征并未使用http://github.com/greencardamom/MediaWikiAwkAPI，而是直接从wikipedia上抓取
抓取程序是 
            Produce_Inlinks.py
直接运行即可。
            python Produce_Inlinks.py
会在当前目录下生成两个inlinks_e.p与inlinks_o.p的文件。
前者是包含entity的点击次数以及entity的字典；后者是包含object的点击次数以及object的字典。

F-CNN的CNN部分与论文的CNN类似。同样需要运行read_xlsx.py来生成词向量。

在执行完以上步骤之后，运行train_F_CNN.py程序即可。
        python train_F_CNN.py
该程序是使用的SVM-L来选取的HCF的特征。不过因为总共得到的特征仅有100+维，因此仅仅是从HCF中选取了100维的特征。