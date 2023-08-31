# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 11:58:55 2023

@author: hohle
"""

#https://jalammar.github.io/illustrated-transformer/
import numpy as np


###############################################################################
class ImportSequence():
    
    def __init__(self, Embedding, CorpusSet):
        
        #Embedding: word embedding eg from Word2Vec
        #           size is Nwords_in_corpus x Ndim_embedding
        #Corpus:    the corpus set(!) the embedding was trained from as a list
        
        #class 
        #
        #- creates a dict based on word embedding, 
        #- reads the input sequence
        #- encodes the inputs sequence based on the embedding
        #- adds positional encoding
        
        
        #some pre processing
        [Nwords_in_corpus , Ndim_embedding] = Embedding.shape
                                              #each chunk is a token
        
        if len(CorpusSet) != Nwords_in_corpus:
            Embedding = Embedding.T#turning matrix the right way, if it is not
        
 
        self.Ndim_embedding = Ndim_embedding
        #######################################################################
        
        #######################################################################
        #creating dictionary, based on embedding and corpus it had been trained on
        TokenToVec = {ch:e for ch,e in zip(CorpusSet,Embedding)}
        Encode     = lambda Sequence: [TokenToVec[c] for c in Sequence]
        
        self.Dict    = TokenToVec
        self.Encoder = Encode 
        
    def encodeSeq(self,Sequence):
        
        #Sequence: sequence of tokens as full string, 
        #i.e. "My God, it's full of stars!"
        
        Sequence = list(Sequence.split())

        #######################################################################
        #encoding according to embedding
        Encode   = self.Encoder
        SeqNum   = np.array(Encode(Sequence)) #size Ntoken x Ndim_embedding
        Ntoken   = SeqNum.shape[0]
        #######################################################################
        
        #######################################################################
        #positional encoding
        n           = 10000
        d           = self.Ndim_embedding
        denominator = np.power(n, 2*np.arange(int(d/2))/d)
        P           = np.zeros((Ntoken,d))
        P[:,::2]    = np.sin(np.arange(Ntoken)/denominator[:,None]).T#odd
        P[:,1::2]   = np.cos(np.arange(Ntoken)/denominator[:,None]).T#even
        #######################################################################

        #adding pos encoding info to encoded seq of token
        SeqNum +=P 

        self.P      = P
        self.SeqNum = SeqNum
        self.Ntoken = Ntoken
        