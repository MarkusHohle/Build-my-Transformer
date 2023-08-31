# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:20:37 2023

@author: hohle
"""

#https://jalammar.github.io/illustrated-transformer/

#https://www.youtube.com/watch?v=_UVfwBqcnbM
#
#note: there is an error at 13:16: ncol(W_0) is not Ntoken!! It is actually
#Ndim_embedding!

#http://www.columbia.edu/~jsl2239/transformers.html
import numpy as np


###############################################################################
class SelfAttentionQKV():
    
    def __init__(self, Ndim_embedding, Nhead = 8):
        
        #######################################################################
        #initializing query, key and value matrices -> to be trained later
        N_QKV = 64 #according to the paper
        
        QKV = np.random.normal(0,1,(Ndim_embedding, N_QKV, Nhead,3))
                
        self.Nhead          = Nhead
        self.Query          = QKV[:,:,:,0]
        self.Key            = QKV[:,:,:,1]
        self.Value          = QKV[:,:,:,2]
        self.N_QKV          = N_QKV
        self.weights        = QKV #naming them weights in order to fit my ANN
        self.dweights       = QKV
        self.Ndim_embedding = Ndim_embedding
        #######################################################################
        

        
    def forward(self, SeqNum):
        #SeqNum: numerical sequence after embedding AND pos encoding from 
        #previous encoder or (if this is the first encoder) from input sequence
       
        Ntoken  = SeqNum.shape[0]
        weights = self.weights
        N_QKV   = self.N_QKV
        Nhead   = self.Nhead
        
        #only have to do once for a new sequence
        if len(SeqNum.shape) == 2:
            SeqNum      = np.repeat(SeqNum[:, :, np.newaxis], Nhead, axis=2)
        
        #sequence times query, key and value
        #'greedy' optimizes sum operations, speeds up process by one magintude
        q = np.einsum('ijk,jmk->imk', SeqNum, weights[:,:,:,0], optimize='greedy')#size Ntoken x N_QKV x Nhead
        k = np.einsum('ijk,jmk->imk', SeqNum, weights[:,:,:,1], optimize='greedy')#size Ntoken x N_QKV x Nhead
        v = np.einsum('ijk,jmk->imk', SeqNum, weights[:,:,:,2], optimize='greedy')#size Ntoken x N_QKV x Nhead
        
        #score
        #still have to figure out how that works with einsum
        s = np.zeros((Ntoken, Ntoken, Nhead))
        for i in range(Nhead):
            s[:,:,i] = np.dot(q[:,:,i],k[:,:,i].T)/N_QKV**0.5#normalizing 
            #scores by length of QKV vectors. Why? because softmax becomes less
            #specific for large values
        
        self.output = s
        self.v      = v
        self.k      = k
        self.q      = q
        self.s      = s
        self.SeqNum = SeqNum
        
    def backward(self, dvalues_s, dvalues_v, P = 1):
        #dvalues_s: score from softmax
        #dvalues_v: from WeightedAver
        #P        : if last encoder (P != 1) or not (P == 1)
        SeqNum  = self.SeqNum
        weights = self.weights
        
        #gradients
        
        if P == 1:
            dq                 = np.einsum('mjk,imk->ijk', self.k, dvalues_s,\
                                                             optimize='greedy')
            dq                 = dq/self.N_QKV**0.5
            
            dk                 = np.einsum('mik,jmk->ijk', self.q, dvalues_s,\
                                                             optimize='greedy')
            dk                 = dk.transpose(1,0,2)
            
            dSeqNum = np.einsum('imk,jmk->jik', weights[:,:,:,0], dq,\
                                                         optimize='greedy') + \
                      np.einsum('imk,jmk->jik', weights[:,:,:,1], dk,\
                                                         optimize='greedy') + \
                      np.einsum('imk,jmk->jik', weights[:,:,:,2], dvalues_v,\
                                                             optimize='greedy')
                          
            self.dweights[:,:,:,0] = np.einsum('mik,mjk->ijk', SeqNum, dq,\
                                                             optimize='greedy')
                
            self.dq = dq
         
        else:
            dk = dvalues_s
            
            dSeqNum = np.einsum('imk,jmk->jik', weights[:,:,:,1], dk,\
                                                         optimize='greedy') + \
                      np.einsum('imk,jmk->jik', weights[:,:,:,2], dvalues_v,\
                                                             optimize='greedy')
        
            
        self.dweights[:,:,:,1] = np.einsum('mik,mjk->ijk', SeqNum, dk,\
                                                             optimize='greedy')
        self.dweights[:,:,:,2] = np.einsum('mik,mjk->ijk', SeqNum, dvalues_v,\
                                                             optimize='greedy')

        self.dk      = dk
        self.dinputs = dSeqNum

###############################################################################

###############################################################################
class CrossAttentionQ():
    
    def __init__(self, Ndim_embedding, Nhead):
        
        #Nhead this time comes from encoder input
        
        #######################################################################
        #initializing query, key and value matrices -> to be trained later
        N_QKV = 64 #according to the paper
        
        Q = np.random.normal(0,1,(Ndim_embedding, N_QKV, Nhead))
                
        self.Nhead          = Nhead
        self.Query          = Q
        self.N_QKV          = N_QKV
        self.weights        = Q #naming them weights in order to fit my ANN
        self.dweights       = Q
        self.Ndim_embedding = Ndim_embedding
        #######################################################################
        

        
    def forward(self, SeqNum, Kencode):
        #Kencode is key from encoder
        #SeqNum: numerical sequence after embedding AND pos encoding from 
        #previous decoder or (if this is the first decoder) from input sequence

        NtokenD = SeqNum.shape[0]  #number of tokens from prev decoder/ seq to be decoded
        NtokenE = Kencode.shape[0] #number of tokens from encoder
        weights = self.weights
        N_QKV   = self.N_QKV
        Nhead   = self.Nhead
        
        #only have to do once for a new sequence
        if len(SeqNum.shape) == 2:
            SeqNum      = np.repeat(SeqNum[:, :, np.newaxis], Nhead, axis=2)
        
        #sequence times query, key and value
        #'greedy' optimizes sum operations, speeds up process by one magintude
        q = np.einsum('ijk,jmk->imk', SeqNum, weights, optimize='greedy')#size Ntoken x N_QKV x Nhead
    
        #score
        #still have to figure out how that works with einsum
        s = np.zeros((NtokenD,NtokenE,Nhead))
        for i in range(Nhead):
            s[:,:,i] = np.dot(q[:,:,i],Kencode[:,:,i].T)/N_QKV**0.5#normalizing 
            #scores by length of QKV vectors. Why? because softmax becomes less
            #specific for large values
        
        self.output = s
        self.q      = q
        self.k      = Kencode
        self.SeqNum = SeqNum
        
    def backward(self, dvalues_s):
        #dvalues_s: score from softmax
        SeqNum  = self.SeqNum
        weights = self.weights
        
        #gradients
        dq            = np.einsum('mjk,imk->ijk', self.k, dvalues_s,\
                                                             optimize='greedy')
        dq            = dq/self.N_QKV**0.5
        
        dk            = np.einsum('mik,mjk->ijk', self.q, dvalues_s,\
                                                             optimize='greedy')
        dk            = dk.transpose(1,0,2)
        
        dSeqNum       = np.einsum('imk,jmk->jik', weights, dq,\
                                                             optimize='greedy')
        self.dweights = np.einsum('mik,mjk->ijk', SeqNum, dq,\
                                                             optimize='greedy')
        
        self.dq      = dq
        self.dk      = dk
        self.dinputs = dSeqNum
        
###############################################################################

###############################################################################
class WeightedAver():
    
    def __init__(self, Ndim_embedding, N_QKV, Nhead):
        
        self.weights   = np.random.normal(0,1,(N_QKV*Nhead,Ndim_embedding))
    
    def forward(self, w, v):
        #differentiation between NtokenE (from encoder) and NtokenD (from
        #decoder), if this class is being called by the decoder function
        
        [NtokenE, N_QKV, Nhead] = v.shape
        NtokenD                 = w.shape[0]
        
        sigma = np.zeros((NtokenD, N_QKV*Nhead))
        
        W = self.weights
        #using weights w to create a weighted lin combination of the value vector 
        #TBD: find a better way than for loops
        #and concatenate them
        for k in range(Nhead):
            for j in range(NtokenD):
                for i in range(NtokenE):
                    sigma[j,N_QKV*k:N_QKV*(k+1)] += w[j,i,k] * v[i,:,k]
        
        
        z = np.dot(sigma,W)
              
        self.output = z
        
        #needed for backprop
        self.sigma  = sigma
        self.N_QKV  = N_QKV
        self.Nhead  = Nhead
        self.v      = v
        self.w      = w
        
    def backward(self, dvalues,*dv):
        #*dv: if class is called by the last encoder and feedback comes
        #from decoder
        v = self.v
        w = self.w
            
        self.dsigma     = np.dot(dvalues, self.weights.T)
        self.dweights   = np.dot(self.sigma.T, dvalues)
            
        #unraveling dsigma
        dsigmaResh      = self.dsigma.reshape(dvalues.shape[0],\
                                           self.N_QKV, self.Nhead, order = 'F')
        self.dsigmaResh = dsigmaResh
            
        #dw[i,j] = v[j,:] * dsigmaResh[i,:] for each k
        self.dw   = np.einsum('jmk,imk->ijk', v, dsigmaResh, optimize='greedy')
        
        if not dv:
            dv = np.zeros(v.shape)
        
            #dv[i,:] = sum_j w[i,j]*dsigma[i,:] 
            for k in range(self.Nhead):
                for j in range(w.shape[1]):#from decoder OR encoder
                    for i in range(w.shape[0]):#range(v.shape[0]):#from encoder
                        dv[i,:,k] += w[i,j,k]*dsigmaResh[i,:,k]
                    
            self.dv   = dv
        else:
            self.dv   = dv[0]
        
###############################################################################

###############################################################################
class AddAndNorm():
    
    def forward(self, z ,SeqNum):
        
        S    = np.sum(z + SeqNum)
        Norm = (z + SeqNum)/S
        Norm -= np.mean(Norm)
        
        self.output = Norm
        self.S      = S
        
    def backward(self, dvalues):
        #gradients
        self.dz      = (1/self.S - 1) * dvalues
        self.dSeqNum = (1/self.S - 1) * dvalues
        

###############################################################################

###############################################################################
class Layer_Dense():
    
    def __init__(self, n_inputs, n_neurons):
        #note: we are using randn here in order to see if neg values are 
        #clipped by the ReLU
        #import numpy as np
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        
#passing on the dot product as input for the next layer, as before
    def forward(self, inputs):
        self.output  = np.dot(inputs, self.weights) + self.biases
        self.inputs  = inputs#we're gonna need for backprop
        
    def backward(self, dvalues):
        #gradients
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        self.dinputs  = np.dot(dvalues, self.weights.T)
        
###############################################################################

###############################################################################
class Activation_ReLU:
    
    def forward(self, inputs):
        self.output  = np.maximum(0,inputs)
        self.inputs  = inputs
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0#ReLU derivative
        

###############################################################################

###############################################################################

class Sigmoid:
        
    def forward(self, M):
        
        sigm        = np.clip(1/(1 + np.exp(-M)), 1e-7, 1 - 1e-7)
        self.output = sigm
        self.inputs = sigm #needed for back prop
            
    def backward(self, dvalues):
        
        sigm         = self.inputs
        deriv        = np.multiply(sigm, (1 - sigm))
        self.dinputs = np.multiply(deriv, dvalues)
        

###############################################################################

###############################################################################
class Activation_Softmax:
  
    def forward(self, inputs):
        self.inputs = inputs
        exp_values  = np.exp(inputs - np.max(inputs, axis = 1,\
                                      keepdims = True))#max in order to 
                                                       #prevent overflow
        #normalizing probs
        probabilities = exp_values/np.sum(exp_values, axis = 1,\
                                      keepdims = True)  
        self.output   = probabilities                                                
    
    def backward(self, dvalues):
        #just initializing a matrix
        self.dinputs = np.empty_like(dvalues)
        
        H = dvalues.shape[2]#for loop over heads --> TBD find a better way
        #to avoid nested loop
        
        for h in range(H):
            for i, (single_output, single_dvalues) in \
                            enumerate(zip(self.output[:,:,h], dvalues[:,:,h])):
            
                single_output       = single_output.reshape(-1,1)
                jacobMatr           = np.diagflat(single_output) - \
                                         np.dot(single_output, single_output.T)
                self.dinputs[:,:,h] = np.dot(jacobMatr, single_dvalues)
###############################################################################

#classes for training evaluation (loss, accuracy)
#--> calls class Activation_Softmax for calculating cross entropy
###############################################################################
class Loss:
     
     def calculate(self, output, y):
         
         sample_losses = self.forward(output, y)
         data_loss     = np.mean(sample_losses)
         return(data_loss)
    
###############################################################################

###############################################################################
class Loss_CategoricalCrossEntropy(Loss): 
                       #y_pred is not the predicted y, it is its 
                       #probability!!
     def forward(self, y_pred, y_true):
         samples = len(y_pred)
         #removing vals close to zero and one bco log and accuracy
         y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
         
         #now, depending on how classes are coded, we need to get the probs
         if len(y_true.shape) == 1:#classes are encoded as [[1],[2],[2],[4]]
             correct_confidences = y_pred_clipped[range(samples), y_true]
         elif len(y_true.shape) == 2:#classes are encoded as
                                    #[[1,0,0], [0,1,0], [0,1,0]]
             correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)
         #now: calculating actual losses
         negative_log_likelihoods = -np.log(correct_confidences)
         return(negative_log_likelihoods)
         
     def backward(self, dvalues, y_true):
         Nsamples = len(dvalues)
         Nlabels  = len(dvalues[0])
         #turning labels into one-hot i. e. [[1,0,0], [0,1,0], [0,1,0]], if
         #they are not
         if len(y_true.shape) == 1:
            #"eye" turns it into a diag matrix, then indexing via the label
            #itself
            y_true = np.eye(Nlabels)[y_true]
         #normalized gradient
         self.dinputs = -y_true/dvalues/Nsamples
        
###############################################################################

###############################################################################
class Activation_Softmax_Loss_CategoricalCrossentropy():
    
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss       = Loss_CategoricalCrossEntropy()
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output#the probabilities
        #calculates and returns mean loss
        return(self.loss.calculate(self.output, y_true))
        
    def backward(self, dvalues, y_true):
        Nsamples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
        self.dinputs = dvalues.copy()
        #calculating normalized gradient
        self.dinputs[range(Nsamples), y_true] -= 1
        self.dinputs = self.dinputs/Nsamples
        
###############################################################################

###############################################################################
class Optimizer_SGD:
    #initializing with a default learning rate of 0.1
    def __init__(self, learning_rate = 1, decay = 0, momentum = 0, L2 = 0):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.momentum              = momentum
        self.L2                    = 2*L2
        
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1/ (1 + self.decay*self.iterations))
        
    def update_params(self, layer):
        
        #if we use momentum
        if self.momentum:
            
            #check if layer has attribute "momentum"
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                
                #attention layers don't have biases:
                if hasattr(layer, 'biases'):
                    layer.bias_momentums   = np.zeros_like(layer.biases)
                
            #now the momentum parts
            weight_updates = self.momentum * layer.weight_momentums - \
                             self.current_learning_rate * layer.dweights + \
                             self.L2 * layer.weight_momentums
            layer.weight_momentums = weight_updates
            
            #attention layers don't have biases:
            if hasattr(layer, 'biases'):
                bias_updates = self.momentum * layer.bias_momentums - \
                               self.current_learning_rate * layer.dbiases + \
                               self.L2 * layer.bias_momentums
                layer.bias_momentums = bias_updates
            
        else:
            
            weight_updates = -self.current_learning_rate * layer.dweights + \
                             self.L2 * layer.weights
            
            #attention layers don't have biases:
            if hasattr(layer, 'biases'):
                bias_updates   = -self.current_learning_rate * layer.dbiases + \
                                 self.L2 * layer.biases
        
        layer.weights += weight_updates
        
        #attention layers don't have biases:
        if hasattr(layer, 'biases'):
            layer.biases  += bias_updates
        
    def post_update_params(self):
        self.iterations += 1













