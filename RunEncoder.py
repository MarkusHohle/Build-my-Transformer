# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:48:21 2023

@author: hohle
"""

import AllModules as All
import numpy as np

class Encoder():
    
    def __init__(self, Ndim_embedding):
        
        QKV        = All.SelfAttentionQKV(Ndim_embedding)
        activation = All.Activation_Softmax()
        AandN1     = All.AddAndNorm()
        AandN2     = All.AddAndNorm()
        FF1        = All.Layer_Dense(Ndim_embedding, 2048)#according to the paper
        FF2        = All.Layer_Dense(2048, Ndim_embedding)#according to the paper
        ReLu       = All.Activation_ReLU()
        #ReLu       = All.Sigmoid()
        WandA      = All.WeightedAver(Ndim_embedding, QKV.N_QKV, QKV.Nhead)
        
        self.QKV            = QKV
        self.activation     = activation
        self.AandN1         = AandN1
        self.AandN2         = AandN2
        self.FF1            = FF1
        self.FF2            = FF2
        self.ReLu           = ReLu
        self.WandA          = WandA
        self.Ndim_embedding = Ndim_embedding

    
    def forward(self, inputs):
        #inputs = SeqNum for 1st encoder
        #inputs = E.output for all other encoders
        
        QKV        = self.QKV
        AandN1     = self.AandN1
        AandN2     = self.AandN2
        FF1        = self.FF1
        FF2        = self.FF2
        ReLu       = self.ReLu
        WandA      = self.WandA
        activation = self.activation

###############################################################################
# self attention
###############################################################################
        QKV.forward(inputs)
        v = QKV.v
        s = QKV.output #S

        
        activation.forward(s)
        w = activation.output

        
        WandA.forward(w,v)
        z = WandA.output
###############################################################################


###############################################################################
# add and norm
###############################################################################
        AandN1.forward(z, inputs)
        forFF = AandN1.output
###############################################################################


###############################################################################
# feed forward
###############################################################################
        FF1.forward(forFF)
        ReLu.forward(FF1.output)
        FF2.forward(ReLu.output)
        fromFF = FF2.output
###############################################################################


###############################################################################
# add and norm
###############################################################################
        AandN2.forward(fromFF, forFF)
        
        
        EncoderOut  = AandN2.output
        
        self.output = EncoderOut
        
###############################################################################

################################################################################
        
    def backward(self, *din):
        #*dvalues, *dk, **dv
        #dvalues from previous encoder, if this is not the last encoder
        #dk      from decoder, if this is the last encoder
        #dv      from decoder, if this is the last encoder
        leArg = len(din)
        
        QKV        = self.QKV
        AandN1     = self.AandN1
        AandN2     = self.AandN2
        FF1        = self.FF1
        FF2        = self.FF2
        ReLu       = self.ReLu
        activation = self.activation
        WandA      = self.WandA
        
        if leArg == 1:#we have one input argument: dvalues
            dvalues = din[0]
            AandN2.backward(dvalues)
            
            FF2.backward(AandN2.dSeqNum)
            ReLu.backward(FF2.dinputs)
            FF1.backward(ReLu.dinputs)
        
            #not sure if adding dz and dinputs is correct!!!
            AandN1.backward(AandN2.dz + FF1.dinputs)
            WandA.backward(AandN1.dz)
            
            dvalues_v = WandA.dv
            
            activation.backward(WandA.dw)
            dvalues_s = activation.dinputs #gradient of score (q*k)
            
            QKV.backward(dvalues_s, dvalues_v)#gradient of key
            
            #also: check here, if adding dz and dinputs is correct!!!
            self.dinputs = np.sum(QKV.dinputs,axis = 2) + AandN1.dSeqNum
            
        else:#we have two input arguments: dk and dv
            dk = din[0]
            dv = din[1]
            QKV.backward(dk, dv, P = 2)
            
            self.dinputs = np.sum(QKV.dinputs,axis = 2)
        
        
        
 
            
        

        
        
        
        
        
        
        
