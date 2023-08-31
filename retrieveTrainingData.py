# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:37:46 2023

@author: MMH_user
"""
#usage:
#Train = retrieveTraingData('TrainEN.txt')

def retrieveTrainingData(filename):
    
    #load training data
    with open(filename,errors="ignore") as f:
        text = f.read()
    text = text.lower()
    #putting f string into list according to line breaks for convenience
    Train = []
    buff  = []
    for c in text:
        if c == '\n':
            Train.append(''.join(buff))
            buff = []
        else:
            buff.append(c)
    else:
        if buff:
           Train.append(''.join(buff))
           
    return(Train)