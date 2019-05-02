### ====== P1 ====================
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('pylab', 'inline')
from scipy import ndimage

data = list()
with open("/Users/januaryshen/Dropbox/S19 - CSE 547/hw2/q1/data/faces.csv") as myfile:
    reader = csv.reader(myfile, delimiter=',')
    for row in reader:
        data.append(row)
        
# the items in data was string, convert to float
data = np.mat(data).astype(np.float)

# calculate the summationMatrix
summationMatrix = data.T.dot(data) / len(data)

# create eigenvalues and eigenvectors
start = time.time()
w, v = np.linalg.eig(summationMatrix)
end = time.time()
print(end - start)

target = list([1,2,10,30,50])

for i in target:
    print(i, w[i-1])

print(sum(w))

# P1b-1
# sum of eigenvalues is 1084.2074349947675
# lambda 1,2,10,30,50 as follows:
# 1 (781.8126992600016+0j)
# 2 (161.15157496732692+0j)
# 10 (3.339586754887817+0j)
# 30 (0.8090877903777284+0j)
# 50 (0.38957773951814617+0j)


# P1b-2
def rec_error(k, eigenvalues):
    return((k, 1 - sum(eigenvalues[0:k])/sum(eigenvalues)))


error_list = list()
for i in range(1, 50+1):
    error_list.append(rec_error(i, w))

plt.plot(*zip(*error_list))
plt.xlabel("Number of iteration")
plt.ylabel("Reconstruction error")
plt.show()

# P1b-3 
# The principle eigenvalue captures the major theme of the pictures, which is the contour of a face. This feature is shared by all images. 
# Other features, such as the shape of eyes and eyebrows, are less commonly shared so the eigenvalues are smaller.

# P1c-1

columns = 5
rows = 2
fig=plt.figure(figsize=(14, 8))

for i in range(1,10 + 1):
    img = v[:,i-1].astype(np.float).reshape(84,96)
    img = ndimage.rotate(img, 270)
    fig.add_subplot(rows, columns, i)
    plt.title("Eigenvector " + str(i))
    plt.imshow(img, cmap = 'gray')
plt.show()

# P1c-2
# 1: blurred image of a face
# 2: contour of a face with light from the right
# 3: contour of a face with light from the back
# 4: contour of a face with light from the front
# 5: contour of a face with light from the left
# 6: contour of a face with light from the top
# 7: contour of a face with lighter scale
# 8: contour of a face with darker scale
# 9: contour of a face with lighter scale
#10: blurred image of a face


#P1d-1
target_image = [1,24,65,68,257] 
k_list = [1,2,5,10,50]

fig=plt.figure(figsize=(14, 16))
columns = 6
rows = 5
counter = 1
for i in target_image:
    img = data[i-1].astype(np.float).reshape(84,96).T
    fig.add_subplot(rows, columns, counter)
    plt.title("Image " + str(i))
    counter += 1
    plt.imshow(img, cmap = 'gray')
    for j in k_list:
        vec = v[:,0:j]
        vec = vec.dot(vec.T)
        img = vec.dot(data[i-1].T).astype(np.float).reshape(84,96).T
        fig.add_subplot(rows, columns, counter)
        plt.title("top k = " + str(j))
        counter += 1
        plt.imshow(img, cmap = 'gray')
plt.show()

### ====== P2 ====================
# P2a
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
            if np.sqrt(np.square(data_m[i]-c1_m[j]).sum()) <= distance_c1:
                distance_c1 = np.sqrt(np.square(data_m[i]-c1_m[j]).sum())
                c1_index = j
            if np.sqrt(np.square(data_m[i]-c2_m[j]).sum()) <= distance_c2:
                distance_c2 = np.sqrt(np.square(data_m[i]-c2_m[j]).sum())
                c2_index = j
        
        cost_euclid_c1 += distance_c1
        cost_euclid_c2 += distance_c2
        cluster_list_c1.append((c1_index, i))
        cluster_list_c2.append((c2_index, i))
    
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
plt.ylabel("Cost of Euclidean distance")
plt.show()


improve_c1 = abs(cost_euclid_list_c1[9][1] - cost_euclid_list_c1[0][1])/cost_euclid_list_c1[9][1]
improve_c2 = abs(cost_euclid_list_c2[9][1] - cost_euclid_list_c2[0][1])/cost_euclid_list_c2[9][1]
print(improve_c1, improve_c2)


# P2b
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


### ====== P3 ====================
import numpy as np
import math
import matplotlib.pyplot as plt

iteration = 40
lambda1 = 0.1
LRate = 0.01
error_list_01 = list()

m = 943
n = 1682
k = 20

P = np.random.rand(n,k)*math.sqrt(5/k)
Q = np.random.rand(m,k)*math.sqrt(5/k)

def PQ_update(vec1, vec2, error, lambda1):
    return(error*vec1 - 2*lambda1*vec2)

for i in range(iteration):
    # update Q and P
    with open("/Users/januaryshen/Dropbox/S19 - CSE 547/hw2/q3/data/ratings.train.txt") as f:
        for line in f:
            movie_id, user_id, rating = [int(i) for i in line.split()]
            error = 2 *(rating - Q[movie_id-1].dot(P[user_id-1]))

            addQ = PQ_update(P[user_id-1], Q[movie_id-1], error, lambda1)
            addP = PQ_update(Q[movie_id-1], P[user_id-1], error, lambda1)

            Q[movie_id-1] = Q[movie_id-1] + LRate*addQ
            P[user_id-1] = P[user_id-1] + LRate*addP

    # calculate total error in this iteration
    error_sum = 0
    with open("/Users/januaryshen/Dropbox/S19 - CSE 547/hw2/q3/data/ratings.train.txt") as f:
        for line in f:
            movie_id, user_id, rating = [int(i) for i in line.split()]
            error_sum += pow(rating - Q[movie_id-1].dot(P[user_id-1]), 2)
        error_sum = error_sum + pow(P,2).sum() + pow(Q,2).sum()
    error_list_01.append(error_sum)

plt.plot(error_list_01, color = "green")
plt.xlabel("Number of iteration")
plt.ylabel("Error")
plt.show()

### ====== P4 ====================
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
AlexU = RecU[499, 0:100]
AlexU_Index = AlexU.argsort()[-5:][::-1]
for i in AlexU_Index:
    print(movies[i], "\n Similarity = %.2f" % AlexU[i])

# item-item
RecI = data.dot(np.divide(1, np.sqrt(Q), where = np.sqrt(Q)!= 0)).dot(data.T).dot(data).dot(np.divide(1, np.sqrt(Q), where = np.sqrt(Q)!= 0))
AlexI = RecI[499, 0:100]
AlexI_Index = AlexI.argsort()[-5:][::-1]
for i in AlexI_Index:
    print(movies[i], "\n Similarity = %.2f" % AlexI[i])
    
  







