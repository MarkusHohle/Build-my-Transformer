# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 19:34:49 2023

@author: MMH_user
"""
import urllib.request
import random
import numpy as np
import MyANN as MyANN
import umap.umap_ as umap #pip install umap-learn
import umap.plot
import matplotlib.pyplot as plt


from OneHot import OneHot
from plotMyEmbedding import plotMyEmbedding


def Word2Vec(epochs = 5, iterations = 5000, window_size = 15, Ndim_embedding = 200):
    
    ###########################################################################
    #reading and preparing corpus
    
    #a) little shakespere
    #my_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    #with urllib.request.urlopen(my_url) as f:
    #   text = f.read().decode('utf-8')
    #
    #b) t
    with open(r'C:\Users\MMH_user\Desktop\QBM\QBM\courses\Python\Language Models\5 modularized\TrainEN.txt', errors="ignore") as f:
        text = f.read()
    
    #c) european parliament
    #with open(r'C:\Users\hohle\Desktop\QBM\courses\Python\Language Models\5 modularized\train sub\TrainDE1.txt', encoding="utf8") as f:
    #    text = f.read()

    C      = text.split()           #actual corpus
    LC     = len(C)                 #length of corpus
    L      = list(set(text.split()))#list of words in corpus
    NClass = len(L)
    

    Vnum   = {ch: i for i, ch in enumerate(L)} #arbitrary enumeration --> later into one hot
    encode = lambda chunk: [Vnum[s] for s in chunk] #encoding function
    ###########################################################################
    
    ###########################################################################
    #initializing shallow NN
    dense1          = MyANN.Layer_Dense(NClass, Ndim_embedding)
    #activation1     = MyANN.Activation_ReLU()
    optimizer       = MyANN.Optimizer_SGD(learning_rate = 0.2, decay = 0.001, momentum = 0.9)
    dense2          = MyANN.Layer_Dense(len(dense1.biases.T), NClass)#
    loss_activation = MyANN.Activation_Softmax_Loss_CategoricalCrossentropy()
    ###########################################################################
    #
    #loading pretrained weights
    #
    dense1.weights = np.load('weights save/weights1.npy')
    dense2.weights = np.load('weights save/weights2.npy')
    #
    dense1.biases = np.load('weights save/bias1.npy')
    dense2.biases = np.load('weights save/bias2.npy')
    #
    ###########################################################################
    
    
    ##for saving & checks purposes only########################################
    V      = encode(L) #turning corpus into numerical vector
    Mall   = OneHot(V, NClass)
    #np.savetxt('weights save/L.txt', L, fmt='%s')

    
    for i in range(iterations):
        
        #selecting random chunks from the corpus of max 1% size (saving RAM!)
        r1 = int(np.random.uniform(0,LC-10*window_size))
        r2 = int(np.random.uniform(window_size, 10*window_size))
    
        Cchunk = C[r1:r1+r2]    #extracting chunk
        V      = encode(Cchunk) #turning chunk into numerical vector
        
        M      = OneHot(V, NClass)#
        Nwords = M.shape[0]
        
        
        for l in range(Nwords - window_size):
            
            #Cbow algorithm
            idx    = int(l + (window_size-1)/2)
            target = M[idx:idx+1,:]
            y      = target
            train  = np.concatenate((M[l:idx,:],M[idx+1:window_size+l,:]))
            
            ##for printing purpose only to check encoding learning
            #idx          = np.argmax(y)
            #current_word = L[idx]
            #print(f' current word:' + current_word +
            #      f' encoding: {idx}')
            ##
            
            for e in range(epochs):
                dense1.forward(train)
                dense2.forward(dense1.output)
                loss_activation.forward(dense2.output, y)
        
                #backward passes
                loss_activation.backward(loss_activation.output, y)
                dense2.backward(loss_activation.dinputs)
                dense1.backward(dense2.dinputs)
        
                optimizer.pre_update_params()#decaying learning rate
                optimizer.update_params(dense1)
                optimizer.update_params(dense2)
                optimizer.post_update_params()#just increasing iteration by one
        
                #print(loss)
        
        if not i % 500:
            
            #creating umap plt in order to visualize learning process
            dense1.forward(M)
            embedded = dense1.output
            
            newXY    = umap.UMAP().fit_transform(embedded)
            plt.scatter(newXY[:, 0],newXY[:, 1], alpha = 0.2)
            plt.gca().set_aspect('equal', 'datalim')
            plt.title('UMAP projection of word embedding, iteration i=' + str(i),\
                  fontsize = 10)
            plt.show()
            
            np.save('weights save/weights1.npy', dense1.weights)
            np.save('weights save/weights2.npy', dense2.weights)
            np.save('weights save/bias1.npy', dense1.biases)
            np.save('weights save/bias2.npy', dense2.biases)
        
        if not i % 1000:
            #saving embedding for complete corpus
            dense1.forward(Mall)
            embedded_all = dense1.output
            np.save('weights save/embedded.npy', embedded_all)
            
            #plotting and saving plots
            plotMyEmbedding(4000,L,embedded_all)
        
    
