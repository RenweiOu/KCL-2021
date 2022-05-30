#--
# solution-to-lab01.py
# Edited by Munkhtulga Battogtokh 11-Dec-2021
# Solutions for "Practical 3: K-means"
# -----------------------------------------------------
# solution-to-practical-3.py
# parsons+martinez-miranda/18-feb-2017
#
# Running k-means on the iris dataset.
#
# Code draws from:
#
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.cluster import KMeans

#
# A function to compute the Euclidian distance between two points.
#
def distance(point1, point2):
    return math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

# Parameters
clusterNum = 3 # number of clusters
error = 0.01   # convergence is no cluster center moving more than this

# Load the data
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

#
# Pick inital cluster centres.
#
# Get max and min values of the features we cluster on
x1_min, x1_max = math.floor(X[:, 0].min()), math.ceil(X[:, 0].max())
x2_min, x2_max =  math.floor(X[:, 1].min()), math.ceil(X[:, 1].max())

# now pick random points in that space.
clusters = []
for i in range(clusterNum):
    center = [random.randrange(x1_min,  x1_max), random.randrange(x2_min,  x2_max)]
    clusters.append(center)

# print initial clsuter centres just in case some of them are duplicates
print ("Initial clusters:")
print (clusters)

#
# Now run K-means.
#
# First build our own version of the data, Xcluster, with an additional element
# that identifies which cluster the point is in. Initlaise that cluster to 0.
Xcluster = []
for i in range(len(X)):
    Xcluster.append([X[i], 0])

stop = 0

while stop == 0:
    # Assign each point to the cluster with the nearest centre.
    for i in range(len(Xcluster)):
        distances = []
        # For each data point, compute the distance to each cluster center
        for j in range(len(clusters)):
            distances.append(distance(Xcluster[i][0], clusters[j]))

        # Then assign the point to the cluster with the nearest center.
        if min(distances[0], distances[1], distances[2]) == distances[0]:
            Xcluster[i][1] = 1
        if min(distances[0], distances[1], distances[2]) == distances[1]:
            Xcluster[i][1] = 2
        if min(distances[0], distances[1], distances[2]) == distances[2]:
            Xcluster[i][1] = 3

    #print Xcluster[:1]

    # Recompute cluster centres.
    x_coord  = [0, 0, 0]
    y_coord  = [0, 0, 0]
    counts   = [0, 0, 0]

    # Sum up x and y coordinate values of points in each cluster.
    for i in range(len(Xcluster)):
        if Xcluster[i][1] == 1:
            x_coord[0] = x_coord[0] + Xcluster[i][0][0]
            y_coord[0] = y_coord[0] + Xcluster[i][0][1]
            counts[0]  = counts[0] + 1
        if Xcluster[i][1] == 2:
            x_coord[1] = x_coord[1] + Xcluster[i][0][0]
            y_coord[1] = y_coord[1] + Xcluster[i][0][1]
            counts[1]  = counts[1] + 1
        if Xcluster[i][1] == 3:
            x_coord[2] = x_coord[2] + Xcluster[i][0][0]
            y_coord[2] = y_coord[2] + Xcluster[i][0][1]
            counts[2]  = counts[2] + 1

    # Use that to compute new cluster centres and how much each centre
    # moves
    diffs = [0, 0, 0]
    for i in range(len(clusters)):
        if counts[i] > 0:
            new_center = [(x_coord[i]/counts[i]), (y_coord[i]/counts[i])]
            diffs[i] = distance(new_center, clusters[i])
            clusters[i] = new_center
   
    # Stop when all centres move less than error
    stop = 1
    for i in range(len(diffs)):
        if diffs[i] > error:
            stop = 0;

# End of while loop

# Extract the cluster identifer into a separate list
labels = []
for i in range(len(Xcluster)):
    labels.append(Xcluster[i][1])

kM_imp = metrics.adjusted_rand_score(y, labels)
print ( "Rand index for k-Means implementation: %.4f" % kM_imp )

# Use sklearn.cluster.KMeans

kM_X = KMeans( n_clusters=3 )
kM_X.fit( X )
kM_labels = kM_X.labels_
kM_sklearn = metrics.adjusted_rand_score(y, kM_labels)
print ("Rand index for k-Means from scikit-learn: %.4f\n" % (kM_sklearn))

# Now compare the results:
if kM_sklearn > kM_imp:
    print ("   Scikit-learn k-Means has a better perfomance\n")
elif kM_sklearn < kM_imp:
    print ("   Your k-Means implementation has a better perfomance\n")
else:
    print ("   Scikit-learn k-Means and your implementation have the same performance\n")

#
# Plot everything
#
plt.subplot( 1, 2, 1 )
# Plot the original data 
plt.scatter(X[:, 0], X[:, 1], c=y.astype(float))
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# title:
plt.title('Raw data')
plt.subplot( 1, 2, 2 )
# Plot the clusters we found.
plt.scatter(X[:, 0], X[:, 1], c=labels)
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# title:
plt.title('k-Means Implementation')
plt.savefig('kMeans_Implem.pdf')
#plt.show()


#
# Plot scikit-learn k-Means
#
plt.subplot( 1, 2, 1 )
# Plot the original data
plt.scatter(X[:, 0], X[:, 1], c=y.astype(float))
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# title:
plt.title('Raw data')
plt.subplot( 1, 2, 2 )
# Plot the clusters we found.
plt.scatter(X[:, 0], X[:, 1], c=kM_labels)
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )
# title:
plt.title('Scikit-learn k-Means')
plt.savefig('kMeans_scikitL.pdf')
#plt.show()




