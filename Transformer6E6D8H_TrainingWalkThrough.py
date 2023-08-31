# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:08:53 2023

@author: MMH_user
"""

n_iter = 100000 
n_epo = 20
learning_rate = 0.1
decay = 0.001
momentum = 0.5
L2 = 0.01)

###############################################################################
#calling libraries
    import numpy as np
    import random
    import re

    import ImportSequence as ImSeq #includes tokenizer, creates dictionary etc
    import RunEncoder     as RunE
    import RunDecoder     as RunD
    
    from retrieveEmbeddingModel import retrieveEmbeddingModel
    from retrieveTrainingData   import retrieveTrainingData

    from AllModules import Layer_Dense as Linear
    from AllModules import Optimizer_SGD 
    from AllModules import Activation_Softmax as SoftMax
    from AllModules import Activation_Softmax_Loss_CategoricalCrossentropy as Entropy
###############################################################################

###############################################################################
#reading corpus and embedding (example: six words and Ndim_embedding = 20)
#Im =  ImSeq.ImportSequence(np.random.normal(0,10,(6,20)),\
#                        ['cat','dog','mouse','tiger','worm','bird'])

#loading embedding vectors and corresponding vocabulary
    #calling embedding from pretrained model, it saves them as 
    #the real 300dim fancy model:
    #retireveEmbeddingModel('glove.6B.300d.txt',300)
    #
    #english embedding
    [Vect, Voc] = retrieveEmbeddingModel('glove.6B.50d.txt',50)
    #german embedding
    Vect_target = np.load('Vect300German.npy')
    Voc_target  = np.load('Vocab300German.npy')
    
    #training set in lower case, since embedding has been trained for lower case
    TrainEN    = retrieveTrainingData('TrainEN.txt')
    LTrain     = len(TrainEN)
    
    fileGerman = 'TrainDE.txt'
    TrainDE    = retrieveTrainingData(fileGerman)
    
    #creating voc corpus of training data for y vector
    with open(fileGerman, errors="ignore") as f:
        text = f.read()
    text     = text.lower()
    ListVocY = list(set(text.split()))
    LV       = len(ListVocY)

    Im             = ImSeq.ImportSequence(Vect,Voc)
    Im_target      = ImSeq.ImportSequence(Vect_target,Voc_target)
    Ndim_embedding = Im.Ndim_embedding

###############################################################################

###############################################################################
#1) transformer ini part
    E1 = RunE.Encoder(Ndim_embedding)
    E2 = RunE.Encoder(Ndim_embedding)
    E3 = RunE.Encoder(Ndim_embedding)
    E4 = RunE.Encoder(Ndim_embedding)
    E5 = RunE.Encoder(Ndim_embedding)
    E6 = RunE.Encoder(Ndim_embedding)

    D1 = RunD.Decoder(Ndim_embedding,E6.QKV.Nhead)
    D2 = RunD.Decoder(Ndim_embedding,E6.QKV.Nhead)
    D3 = RunD.Decoder(Ndim_embedding,E6.QKV.Nhead)
    D4 = RunD.Decoder(Ndim_embedding,E6.QKV.Nhead)
    D5 = RunD.Decoder(Ndim_embedding,E6.QKV.Nhead)
    D6 = RunD.Decoder(Ndim_embedding,E6.QKV.Nhead)

    L = Linear(Ndim_embedding, LV)#need a prediction for every token in the
                                  #dictionary
    S = SoftMax()

    E = Entropy()
    
    optimizer = Optimizer_SGD(learning_rate, decay, momentum, L2)
###############################################################################

###############################################################################
#starting training loop
    for i in range(n_iter):
        #picking random training phrase
        idx    = random.randint(0,LTrain-1)
        train  = TrainEN[idx]
        target = TrainDE[idx]
        
       
        #encoding sequence
        Im.encodeSeq(train)
        Im_target.encodeSeq(target)
        SeqNum = Im.SeqNum
        Ntoken = Im_target.Ntoken
        
        #creating one hot matrix for target vector.
        #note: one hot for entire vocab is too large and crashes computer
        #--> need to do this only for specific tokens each time in the loop
        Idx    = [ListVocY.index(s) for s in target.split()]
        y      = np.zeros((Ntoken+2,LV))# +2 because of SOS and EOS token
        for i, idx in enumerate(Idx):
            y[i+1,idx] = 1
            
        #also here: proper SOS and EOS token from embedding!!
        y[0,97]  = 1# SOS token
        y[-1,97] = 1# EOS token
        
        
        for e in range(n_epo):
            #starting training on part sequence
            #2)transformer forward part

            #2a) encoder
            E1.forward(SeqNum)
            E2.forward(E1.output)
            E3.forward(E2.output)
            E4.forward(E3.output)
            E5.forward(E4.output)
            E6.forward(E5.output)
                
            for n in range(Ntoken+2):#last token is EOS token

                #2b) decoder
                #decoder gets key and value from encoder!
                #https://datascience.stackexchange.com/questions/51785/what-is-the-first-input-to-the-decoder-in-a-transformer-model
                if n == 0:
                    #first input for decoder is a neutral token (here _)
                    SeqNumDec = Vect[97].reshape(1,Ndim_embedding)
                else:
                    SeqNumDec = np.vstack((SeqNumDec, D6.output[-1,:]))
          
                D1.forward(SeqNumDec,E6.QKV.k,E6.QKV.v)
                D2.forward(D1.output,E6.QKV.k,E6.QKV.v)
                D3.forward(D2.output,E6.QKV.k,E6.QKV.v)
                D4.forward(D3.output,E6.QKV.k,E6.QKV.v)
                D5.forward(D4.output,E6.QKV.k,E6.QKV.v)
                D6.forward(D5.output,E6.QKV.k,E6.QKV.v)

            #2c) "tail end"
            L.forward(D6.output)
            
            S.forward(L.output)
###############################################################################

###############################################################################
            #3) evaluation
            #S.output == E.output --> just for testing
            loss            = E.forward(L.output, y)
            #predictions     = np.argmax(E.output, axis = 1)
            predictions     = np.argmax(S.output, axis = 1)

            #if len(y.shape) == 2:
            #    y = np.argmax(y,axis = 1)
                    
            #accuracy = np.mean(predictions == y)


###############################################################################

###############################################################################
            #3) transformer backward part
            E.backward(E.output, y)
            L.backward(E.dinputs)# we can skip S here, because S.output == E.output
        
            D6.backward(L.dinputs)
            D5.backward(D6.dinputs)
            D4.backward(D5.dinputs)
            D3.backward(D4.dinputs)
            D2.backward(D3.dinputs)
            D1.backward(D2.dinputs)
        
            dK6 = D6.dKencode #backprop for key from decoder 1
            dK5 = D5.dKencode 
            dK4 = D4.dKencode 
            dK3 = D3.dKencode 
            dK2 = D2.dKencode 
            dK1 = D1.dKencode 
        
            dV6 = D6.dVencode
            dV5 = D5.dVencode
            dV4 = D4.dVencode
            dV3 = D3.dVencode
            dV2 = D2.dVencode
            dV1 = D1.dVencode
        
            dK = dK1 + dK2 + dK3 + dK4 + dK5 + dK6
            dV = dV1 + dV2 + dV3 + dV4 + dV5 + dV6
        
            E6.backward(dK, dV)
            E5.backward(E6.dinputs)
            E4.backward(E5.dinputs)
            E3.backward(E4.dinputs)
            E2.backward(E3.dinputs)
            E1.backward(E2.dinputs)

###############################################################################

###############################################################################
#4) transformer updating weights
        
            #note: optimizer looks for the specific keywords 
            #"weights" and "dweights" in layers. Order not important

            optimizer.pre_update_params()#decaying learning rate
        
            #tail end dense layer
            optimizer.update_params(L)
        
            #decoder self attention updating query, key, value
            optimizer.update_params(D1.QKV)
            optimizer.update_params(D2.QKV)
            optimizer.update_params(D3.QKV)
            optimizer.update_params(D4.QKV)
            optimizer.update_params(D5.QKV)
            optimizer.update_params(D6.QKV)

            #decoder dense layer weights 
            optimizer.update_params(D1.FF1)
            optimizer.update_params(D2.FF1)
            optimizer.update_params(D3.FF1)
            optimizer.update_params(D4.FF1)
            optimizer.update_params(D5.FF1)
            optimizer.update_params(D6.FF1)

            optimizer.update_params(D1.FF2)
            optimizer.update_params(D2.FF2)
            optimizer.update_params(D3.FF2)
            optimizer.update_params(D4.FF2)
            optimizer.update_params(D5.FF2)
            optimizer.update_params(D6.FF2)

            #decoder weighted average weights 
            optimizer.update_params(D1.WandA1)
            optimizer.update_params(D2.WandA1)
            optimizer.update_params(D3.WandA1)
            optimizer.update_params(D4.WandA1)
            optimizer.update_params(D5.WandA1)
            optimizer.update_params(D6.WandA1)

            optimizer.update_params(D1.WandA2)
            optimizer.update_params(D2.WandA2)
            optimizer.update_params(D3.WandA2)
            optimizer.update_params(D4.WandA2)
            optimizer.update_params(D5.WandA2)
            optimizer.update_params(D6.WandA2)

            #decoder cross attention 
            optimizer.update_params(D1.CAQ)
            optimizer.update_params(D2.CAQ)
            optimizer.update_params(D3.CAQ)
            optimizer.update_params(D4.CAQ)
            optimizer.update_params(D5.CAQ)
            optimizer.update_params(D6.CAQ)

        
            #encoder self attention
            optimizer.update_params(E1.QKV)
            optimizer.update_params(E2.QKV)
            optimizer.update_params(E3.QKV)
            optimizer.update_params(E4.QKV)
            optimizer.update_params(E5.QKV)
            optimizer.update_params(E6.QKV)
        
            #encoder dense layer weights 
            optimizer.update_params(E1.FF1)
            optimizer.update_params(E2.FF1)
            optimizer.update_params(E3.FF1)
            optimizer.update_params(E4.FF1)
            optimizer.update_params(E5.FF1)
            #optimizer.update_params(E6.FF1)   #not needed, since last encoder
                                               #only delivers keys and values
            optimizer.update_params(E1.FF2)
            optimizer.update_params(E2.FF2)
            optimizer.update_params(E3.FF2)
            optimizer.update_params(E4.FF2)
            optimizer.update_params(E5.FF2)
            #optimizer.update_params(E6.FF2)   #not needed, since last encoder
                                               #only delivers keys and values
            #decoder weighted average weights 
            optimizer.update_params(E1.WandA)
            optimizer.update_params(E2.WandA)
            optimizer.update_params(E3.WandA)
            optimizer.update_params(E4.WandA)
            optimizer.update_params(E5.WandA)
            #optimizer.update_params(E6.WandA) #not needed, since last encoder
                                               #only delivers keys and values

            optimizer.post_update_params()#just increasing iteration by one
