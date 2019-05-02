#!/usr/bin/env python
# coding: utf-8

import re
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('pylab', 'inline')

conf = SparkConf()
sc = SparkContext(conf=conf)

data_m = np.matrix(np.loadtxt("/Users/januaryshen/Dropbox/S19 - CSE 547/hw2/q2/data/data.txt"), dtype = float)
c1_m = np.matrix(np.loadtxt("/Users/januaryshen/Dropbox/S19 - CSE 547/hw2/q2/data/c1.txt"), dtype = float)
c2_m = np.matrix(np.loadtxt("/Users/januaryshen/Dropbox/S19 - CSE 547/hw2/q2/data/c2.txt"), dtype = float)

MAX_ITER = 20
cost_euclid_list_c1 = list()
cost_euclid_list_c2 = list()

for n in range(MAX_ITER):
    cost_euclid_c1 = 0
    cost_euclid_c2 = 0
    cluster_list_c1 = list()
    cluster_list_c2 = list()

    # get the closest centroid
    for i in range(len(data_m)):
        distance_c1 = float("inf")
        distance_c2 = float("inf")
        
        for j in range(len(c1_m)):
            if abs(data_m[i]-c1_m[j]).sum() <= distance_c1:
                distance_c1 = abs(data_m[i]-c1_m[j]).sum()
                c1_index = j
            if abs(data_m[i]-c2_m[j]).sum() <= distance_c2:
                distance_c2 = abs(data_m[i]-c2_m[j]).sum()
                c2_index = j
        
        cost_euclid_c1 += distance_c1
        cost_euclid_c2 += distance_c2
        cluster_list_c1.append((c1_index, i))
        cluster_list_c2.append((c2_index, i))
    
    cost_euclid_list_c1.append((n + 1, cost_euclid_c1))
    cost_euclid_list_c2.append((n + 1, cost_euclid_c2))
    
    cluster_listc1_rdd = sc.parallelize(cluster_list_c1)
    cluster_listc2_rdd = sc.parallelize(cluster_list_c2)

    # cluster all the points that are assigned to the same clostroid
    c1_index = cluster_listc1_rdd.groupByKey().mapValues(list).collect()
    c2_index = cluster_listc2_rdd.groupByKey().mapValues(list).collect()
    
    for k in range(10):
        c1_m[k] = np.mean(data_m[c1_index[k][1]], axis=0)
        c2_m[k] = np.mean(data_m[c2_index[k][1]], axis=0)

    # calculate the cost in each iteration of all items in data
    cost_euclid_list_c1.append((n + 1, cost_euclid_c1))
    cost_euclid_list_c2.append((n + 1, cost_euclid_c2))


plt.plot(*zip(*cost_euclid_list_c1), label='initial cluster centroids = c1')
plt.plot(*zip(*cost_euclid_list_c2), label='initial cluster centroids = c2')
plt.legend()
plt.xlabel("Number of iteration")
plt.ylabel("Cost of Manhattan distance")
plt.show()


improve_c1 = abs(cost_euclid_list_c1[9][1] - cost_euclid_list_c1[0][1])/cost_euclid_list_c1[9][1]
improve_c2 = abs(cost_euclid_list_c2[9][1] - cost_euclid_list_c2[0][1])/cost_euclid_list_c2[9][1]
print(improve_c1, improve_c2)

