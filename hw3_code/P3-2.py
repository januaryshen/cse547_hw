#!/usr/bin/env python
# coding: utf-8

import re
import sys
from pyspark import SparkConf, SparkContext
import numpy as np

conf = SparkConf().set("spark.driver.memory","8G")
sc = SparkContext(conf=conf)

rdd_file = sc.textFile('/home/januaryshen/hw3/graph-full.txt')
rdd_file = rdd_file.map(lambda l: re.split(r'[^\w]+', l))

beta = 0.8
n = 1000
l = np.repeat(float(1), n)
r = 1/n*l
new_r = np.repeat(0.0, n)


# Remove repetitive tuples
# https://stackoverflow.com/questions/48994810/remove-duplicate-tuples-in-an-rdd-in-python
rdd_file = rdd_file.map(lambda x: tuple(x)).distinct()


def f(x): return len(x)
node_weight_list = sorted(rdd_file.groupByKey().mapValues(list).mapValues(f).collect(), key=lambda x: int(x[0]), reverse=False) # list[0] is the node, list[1] is the number of outlet
position_rdd = rdd_file.groupBy(lambda tup: tup[1]).mapValues(list)

for i in range(40):
    for j in range(n):
        row_target = position_rdd.filter(lambda x: int(x[0]) == j+1).collect()[0][1]
        new_r_weight = 0
        for k in range(len(row_target)):
            col = int(row_target[k][0])
            weight = node_weight_list[col-1][1]
            new_r_weight += (1/weight*r[col-1])
        new_r[j] = new_r_weight
    r = (1-beta)/n * l + beta * new_r
print(sum(r))


# return the top 5 indexes
top5 = r.argsort()[-5:][::-1]
count = 1
for i in top5:
    print("The top %d Node is" % count, i+1, "with PageRank score of %.6f" % r[i])
    count += 1


# In[ ]:


# return the buttom 5 indexes
buttom5 = np.argpartition(r, 5)[0:5]
count = 1
for i in buttom5:
    print("The buttom %d Node is" % count, i+1, "with PageRank score of %.8f" % r[i])
    count += 1


