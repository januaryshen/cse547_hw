#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import time


# In[2]:


conf = SparkConf().set("spark.driver.memory","8G")
sc = SparkContext(conf=conf)


# In[3]:


rdd_file = sc.textFile('/Users/januaryshen/Dropbox/S19 - CSE 547/hw3/q2/data/graph-full.txt')
rdd_file = rdd_file.map(lambda l: re.split(r'[^\w]+', l))


# In[4]:


n = 1000
lamb = 1
mu = 1
l = np.repeat(float(1), n)
h = np.repeat(float(1), n)
a = np.repeat(float(1), n)


# In[5]:


# Remove repetitive tuples
# https://stackoverflow.com/questions/48994810/remove-duplicate-tuples-in-an-rdd-in-python
rdd_file = rdd_file.map(lambda x: tuple(x)).distinct()
L_rdd = rdd_file.map(lambda x: (x, 1))


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[ ]:




