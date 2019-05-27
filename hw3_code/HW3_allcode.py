# ======== HW3 P2-1 ===========

import re
import sys
from pyspark import SparkConf, SparkContext
import numpy as np

conf = SparkConf().set("spark.driver.memory","8G")
sc = SparkContext(conf=conf)
rdd_file = sc.textFile('/home/januaryshen/hw3/graph-full.txt')

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

# return the buttom 5 indexes
buttom5 = np.argpartition(r, 5)[0:5]
count = 1
for i in buttom5:
    print("The buttom %d Node is" % count, i+1, "with PageRank score of %.8f" % r[i])
    count += 1


# ======== HW3 P2-2 ===========

import re
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import time

conf = SparkConf().set("spark.driver.memory","8G")
sc = SparkContext(conf=conf)
rdd_file = sc.textFile('/Users/januaryshen/Dropbox/S19 - CSE 547/hw3/q2/data/graph-full.txt')

n = 1000
lamb = 1
mu = 1
l = np.repeat(float(1), n)
h = np.repeat(float(1), n)
a = np.repeat(float(1), n)

# Remove repetitive tuples
# https://stackoverflow.com/questions/48994810/remove-duplicate-tuples-in-an-rdd-in-python
rdd_file = rdd_file.map(lambda x: tuple(x)).distinct()
L_rdd = rdd_file.map(lambda x: (x, 1))

start = time.time()
for k in range(40):    
    for i in range(n):
        a[i] = len(L_rdd.filter(lambda x: int(x[0][1]) == i+1).map(lambda x: (int(x[0][0]), 1)).collect())
    a = mu * a/max(a)

    for j in range(n):
        h[j] = len(L_rdd.filter(lambda x: int(x[0][0]) == j+1).map(lambda x: (int(x[0][1]), 1)).collect())
    h = lamb * h/max(h)

end = time.time()
print(end-start)

# return the top 5 indexes for hubbiness
top5 = h.argsort()[-5:][::-1]
count = 1
for i in top5:
    print("The top %d Node is" % count, i+1, "with hubbiness score of %.8f" % h[i])
    count += 1
    
# return the bottom 5 indexes for hubbiness
bottom5 = np.argpartition(h, 5)[0:5]
count = 1
for i in bottom5:
    print("The bottom %d Node is" % count, i+1, "with hubbiness score of %.8f" % h[i])
    count += 1

# return the top 5 indexes for authority
top5 = a.argsort()[-5:][::-1]
count = 1
for i in top5:
    print("The top %d Node is" % count, i+1, "with authority score of %.8f" % a[i])
    count += 1
    
# return the bottom 5 indexes for authority
bottom5 = np.argpartition(a, 5)[0:5]
count = 1
for i in bottom5:
    print("The bottom %d Node is" % count, i+1, "with authority score of %.8f" % a[i])
    count += 1




