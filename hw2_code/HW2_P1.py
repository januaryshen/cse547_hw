#!/usr/bin/env python
# coding: utf-8

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
