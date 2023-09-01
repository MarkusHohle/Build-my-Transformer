# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:16:40 2023

@author: MMH_user
"""

import numpy as np
import umap.umap_ as umap #pip install umap-learn
import umap.plot
import matplotlib.pyplot as plt
import hdbscan

from random import sample


def plotMyEmbedding(Nmax,L,embedded):

    #embedded = np.load('weights save/embedded.npy')
    
    #with open('L.txt') as f:
    #    L= f.read()
    #L = L.split()

#umap crashes if full set is beeing load at once --> doing it stepwise

    [nr, nc]   = embedded.shape
    subsetsize = Nmax
    rat        = int(np.floor(nr/subsetsize))
    newXY      = np.zeros((nr,2))

    for i in range(rat):
    #making sure plots are aligned, since UMAP is stochastic
         newXY[i*(subsetsize+1):subsetsize*(i+1),:] = umap.UMAP(random_state = 123).fit_transform(\
         embedded[i*(subsetsize+1):subsetsize*(i+1),:])
        
    newXY[rat*subsetsize:-1,:] = umap.UMAP(random_state = 123).fit_transform(\
                             embedded[rat*subsetsize:-1,:])
        
    #random sampling for labels
    np.random.seed()
    idx = sample(range(nr),50)
    x   = newXY[idx, 0]
    y   = newXY[idx, 1]


    plt.scatter(newXY[:, 0],newXY[:, 1], alpha = 0.002, c = 'k')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of final word embedding',fontsize = 10)
    plt.savefig('UMAP Shakespere.jpg', format='jpg', dpi=1200)
    #plt.savefig('UMAP Shakespere.eps', format='eps')
    plt.savefig('UMAP Shakespere.pdf')
    plt.show()
    
    clusterer  = hdbscan.HDBSCAN(min_cluster_size=30).fit(newXY)
    plt.scatter(newXY[:,0], newXY[:, 1], s=20, linewidth=0, c=clusterer.labels_, alpha=0.005)
    plt.title('HDBSCAN clustering',fontsize = 10)
    for i in range(len(x)):
        plt.annotate(L[i], (x[i], y[i] + 0.2), fontsize = 5)
    plt.savefig('UMAP Shakespere HDBscan.jpg', format='jpg',dpi=2200)
    #plt.savefig('UMAP Shakespere HDBscan.eps', format='eps')
    plt.savefig('UMAP Shakespere HDBscan.pdf')
    plt.show()









