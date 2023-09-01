# Build-my-Transformer
Lecture material for building a transformer from scratch (numpy)

Note 1: Material is preliminary, but codes work : )
Note 2: Since the idea is to teach how the very core of a transformer works, tensor flow is not used. Therefore, tensor multiplications are performed with Einstein summation explicitly. 

Repository contains the following libraries:

# AllModules.py    
contains all standar classes incl backprob for building a transformer:
  -  dense layer
  -  self attention and cross attention (default is eight heads, N = 64 dimension for key, value and query matrices is hardcoded)
  -  optimizers (L2, momentum etc)
  -  common activation functions
  -  loss and cross entropy calculator
  -  weight & average
  -  add & norm


# RunDecoder.py  &  RunEncoder.py
Examples how to construct encoder and decoder from "AllModules.py" including backprob


# auxillaries

 - ImportSequence.py (loading recored speeches from the European parliament in German & English as example)
 - retrieveEmbeddingModel.py (loading external embedding model, note: should contain identical set of tokens as loaded by ImportSequence.py)


# example for building a transformer
Note: just an example, not ready to be trained
  - Transformer6E6D8H_TrainingWalkThrough.py
 
