#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math

data = np.loadtxt("/Users/januaryshen/Dropbox/S19 - CSE 547/hw2/q4/data/user-shows.txt")
with open("/Users/januaryshen/Dropbox/S19 - CSE 547/hw2/q4/data/shows.txt") as f:
    movies = f.read().splitlines()

m = data.shape[0]
n = data.shape[1]

P = np.diag(np.diag(data.dot(data.T)))
Q = np.diag(np.diag(data.T.dot(data)))

# user-user
RecU = np.divide(1, np.sqrt(P), where = np.sqrt(P)!= 0).dot(data).dot(data.T).dot(np.divide(1, np.sqrt(P), where = np.sqrt(P)!= 0)).dot(data)
AlexU = RecU[499, 0:99]
AlexU_Index = AlexU.argsort()[-5:][::-1]
for i in AlexU_Index:
    print(movies[i], "\n Similarity = %.2f" % AlexU[i])

# item-item
RecI = data.dot(np.divide(1, np.sqrt(Q), where = np.sqrt(Q)!= 0)).dot(data.T).dot(data).dot(np.divide(1, np.sqrt(Q), where = np.sqrt(Q)!= 0))
AlexI = RecI[499, 0:99]
AlexI_Index = AlexI.argsort()[-5:][::-1]
for i in AlexI_Index:
    print(movies[i], "\n Similarity = %.2f" % AlexI[i])
    
  
