# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 11:28:40 2023

@author: MMH_user
"""
#usage:
#retireveEmbeddingModel('glove.6B.50d.txt',50)
#retireveEmbeddingModel('glove.6B.100d.txt',100)
#retireveEmbeddingModel('glove.6B.200d.txt',200)
#retireveEmbeddingModel('glove.6B.300d.txt',300)
#
#

import numpy as np

def retrieveEmbeddingModel(filename,dim):

    with open(filename,errors="ignore") as f:
        text = f.read()


    out = []
    buff = []
    for c in text:
        if c == '\n':
            out.append(''.join(buff))
            buff = []
        else:
            buff.append(c)
    else:
        if buff:
           out.append(''.join(buff))



    sample_list=list(range(len(out)))
    vector_list=np.zeros((len(out),dim))

   
    
    for ct, sublist in enumerate(out):
    
       c1 = 0
       c2 = 0
       for s in sublist:

           if s.find(' ') == -1 and c2 ==0:
               c1 += 1
            
           if s.find(' ') == 0 and c2 == 0:
                c2 = 1
                sample_list[ct] = sublist[0:c1]
                L = sublist[c1:len(sublist)].split()
            
            
                for num, l in enumerate(L):
                    L[num] = float(l)
            
                vector_list[ct,:] = L
            
        
    #np.save('Vect.npy', vector_list)
    #np.save('Vocab.npy', sample_list)
        
    return(vector_list,sample_list)