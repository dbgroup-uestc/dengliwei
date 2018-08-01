#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:24:20 2018

@author: dlw
"""
from bs4 import BeautifulSoup
import urllib
from collections import defaultdict
import cPickle

if __name__== '__main__':
    print "loading data..."
    x = cPickle.load(open("mr.p","rb"))		#加载之前处理好的数据
    w2v, revs = x[0], x[1]					#数据还原
    print "data loaded!"
    
    #得到entity、predicate、object的字典；其value就是该word出现的次数
    entity_dict = defaultdict(int)
    predicate_dict = defaultdict(int)
    object_dict= defaultdict(int)
    
    for rev in revs:
        entity_dict[rev['Triple'][0]] += 1
        predicate_dict[" ".join(rev['Triple'][1])] += 1
        object_dict[rev['Triple'][2]] += 1
    
    #https://en.wikipedia.org/w/index.php?title=Aamir_Khan&action=info
    urls_entity = []
    urls_object = []
    inlinks_entity = []
    inlinks_object = []
    i = 1
    for entity in entity_dict:
        urls_entity.append("https://en.wikipedia.org/w/index.php?title=%s&action=info" % entity)
    for obj in object_dict:
        urls_object.append("https://en.wikipedia.org/w/index.php?title=%s&action=info" % obj)
#    for url in urls_entity:
#        page = urllib.urlopen(url)
#        Soup = BeautifulSoup(page, 'lxml')
#        num_tag = Soup.select('.mw-pvi-month')
#        if num_tag == []:
#            inlinks_entity.append(int(0))
#        else:
#            inlinks_entity.append(int(num_tag[0].get_text().replace(',','')))
#        print '%d turn.' % i
#        i += 1
#    
#    cPickle.dump([inlinks_entity, entity_dict], open('inlinks_e.p', 'wb'))
    
    i = 0
    for url in urls_object:
        page = urllib.urlopen(url)
        Soup = BeautifulSoup(page, 'lxml')
        num_tag = Soup.select('.mw-pvi-month')
        if num_tag == []:
            inlinks_object.append(int(0))
        else:
            inlinks_object.append(int(num_tag[0].get_text().replace(',','')))
        print '%d turn.' % i
        i += 1
    cPickle.dump([inlinks_object, object_dict], open('inlinks_o.p', 'wb'))