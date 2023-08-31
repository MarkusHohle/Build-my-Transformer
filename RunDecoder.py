# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:00:33 2023

@author: hohle
"""
# almost same structure as encoder
import AllModules as All
import numpy as np

class Decoder():

    def __init__(self, Ndim_embedding, Nhead):

        QKV         = All.SelfAttentionQKV(Ndim_embedding)
        CAQ         = All.CrossAttentionQ(Ndim_embedding, Nhead)
        activation1 = All.Activation_Softmax()
        activation2 = All.Activation_Softmax()
        AandN1      = All.AddAndNorm()
        AandN2      = All.AddAndNorm()
        AandN3      = All.AddAndNorm()
        FF1         = All.Layer_Dense(Ndim_embedding, 2048)  # according to the paper
        FF2         = All.Layer_Dense(2048, Ndim_embedding)  # according to the paper
        ReLu        = All.Activation_ReLU()
        #ReLu        = All.Sigmoid()
        WandA1      = All.WeightedAver(Ndim_embedding, QKV.N_QKV, QKV.Nhead)
        WandA2      = All.WeightedAver(Ndim_embedding, QKV.N_QKV, QKV.Nhead)

        self.QKV             = QKV
        self.CAQ             = CAQ
        self.activation1     = activation1
        self.activation2     = activation2
        self.AandN1          = AandN1
        self.AandN2          = AandN2
        self.AandN3          = AandN3
        self.FF1             = FF1
        self.FF2             = FF2
        self.ReLu            = ReLu
        self.Nhead           = QKV.Nhead
        self.WandA1          = WandA1
        self.WandA2          = WandA2
        self.Ndim_embedding  = Ndim_embedding


    def forward(self, inputs, Kencode, Vencode):
        # inputs = SeqNum for 1st decoder
        # inputs = D.output for all other decoders

        # Kencode: key from encoder for encoder - decoder attention
        # Vencode: value from encoder for encoder - decoder attention

        CAQ         = self.CAQ
        QKV         = self.QKV
        AandN1      = self.AandN1
        AandN2      = self.AandN2
        AandN3      = self.AandN3
        FF1         = self.FF1
        FF2         = self.FF2
        ReLu        = self.ReLu
        WandA1      = self.WandA1
        WandA2      = self.WandA2
        activation1 = self.activation1
        activation2 = self.activation2

###############################################################################
# self attention
###############################################################################
        QKV.forward(inputs)
        v  = QKV.v
        s1 = QKV.output  # S
        
        mask = np.zeros(s1.shape[0:2]) - 1e+100
        mask = np.triu(mask,s1.shape[0]-1)
        mask = np.repeat(mask[:, :, np.newaxis], s1.shape[2], axis=2)

        activation1.forward(s1 + mask)
        w1 = activation1.output

        WandA1.forward(w1, v)
        z1 = WandA1.output
###############################################################################


###############################################################################
# add and norm
###############################################################################
        AandN1.forward(z1, inputs)
        forEDA = AandN1.output
###############################################################################


###############################################################################
# Encoder - Decoder Attention (EDA)
###############################################################################
        CAQ.forward(forEDA, Kencode)# Kencode are keys from encoder
        s2 = CAQ.output  # S

        activation2.forward(s2)
        w2 = activation2.output
        # Vencode are values from encoder
        WandA2.forward(w2, Vencode)
        z2 = WandA2.output
        
###############################################################################
# add and norm
###############################################################################
        AandN2.forward(z2, forEDA)
        forFF = AandN2.output
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
        AandN3.forward(fromFF, forFF)

        DecoderOut  = AandN3.output

        self.output = DecoderOut
        
    def backward(self, dvalues):
        
        CAQ         = self.CAQ
        QKV         = self.QKV
        AandN1      = self.AandN1
        AandN2      = self.AandN2
        AandN3      = self.AandN3
        FF1         = self.FF1
        FF2         = self.FF2
        ReLu        = self.ReLu
        activation1 = self.activation1
        activation2 = self.activation2
        WandA1      = self.WandA1
        WandA2      = self.WandA2
        
        AandN3.backward(dvalues)
        
        FF2.backward(AandN3.dSeqNum)
        ReLu.backward(FF2.dinputs)
        FF1.backward(ReLu.dinputs)
        
        AandN2.backward(AandN3.dz + FF1.dinputs)
        
        WandA2.backward(AandN2.dz)
        
        self.dVencode = WandA2.dv#dVencode --> has to go back to last encoder
        
        activation2.backward(WandA2.dw)
        
        CAQ.backward(activation2.dinputs)
        self.dKencode = CAQ.dk#dKencode --> has to go back to last encoder
        
        AandN1.backward(AandN2.dSeqNum + np.sum(CAQ.dinputs,axis = 2))
        
        WandA1.backward(AandN1.dz)
        
        activation1.backward(WandA1.dw)
        
        QKV.backward(activation1.dinputs, WandA1.dv)
        
        #finally, overall Dncoder gradient
        #also: check here, if adding dz and dinputs is correct!!!
        self.dinputs = np.sum(QKV.dinputs,axis = 2) + AandN1.dSeqNum
        
        
        
        
        
        
