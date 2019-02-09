import random
import math
import numpy as np
from matplotlib import pyplot as plt

class kmeans(object):
    # Distance between two points
    def pdist(self,p0,p1):
        dist = 0.0
        for i in range(0,len(p0)):
            dist += (p0[i] - p1[i])**2
        return math.sqrt(dist)

    def one_hot(self,x, n):
        if type(x) == list:
            x = np.array(x)
        x = x.flatten()
        o_h = np.zeros((len(x), n))
        o_h[np.arange(len(x)), x.astype(int)] = 1
        return o_h

    def calcsse(self,points,cluster,centres,k):
        oh = self.one_hot(cluster,k+1)
        sse = 0
        for p in range(len(points)):
            sse += np.dot(oh[p],(points[p]-centres)**2)
        return sum(sse)/len(points)


    def train(self,k,datapoints):

        d = len(datapoints[0]) 

        Max_Iterations = 1000

        cluster = [0] * len(datapoints)
        prev_cluster = [-1] * len(datapoints)

        cluster_centers = []
        for i in range(0,k):
            cluster_centers += [random.choice(datapoints)]
        i = 0
        sse = []
        while (cluster != prev_cluster) or (i > Max_Iterations):

            prev_cluster = list(cluster)
            i += 1
            for p in range(0,len(datapoints)):
                min_dist = float("inf")
                for c in range(0,len(cluster_centers)):
                    dist = self.pdist(datapoints[p],cluster_centers[c])    
                    if (dist < min_dist):
                        min_dist = dist  
                        cluster[p] = c 

            for k in range(0,len(cluster_centers)):
                new_center = [0] * d
                count = 0
                for p in range(0,len(datapoints)):
                    if (cluster[p] == k):
                        for j in range(0,d):
                            new_center[j] += datapoints[p][j]
                        count += 1

                for j in range(0,d):
                    new_center[j] = new_center[j] / float(count)          
                cluster_centers[k] = new_center
            sse.append(self.calcsse(datapoints,cluster,cluster_centers,k))

#         print("Clusters", cluster_centers)
        print("Total number of Iterations",i)

        return cluster_centers, cluster,sse,i

def getParams(X,k):
    kms = kmeans()
    centres, clusters, sse,n = kms.train(k,X)
    return centres
    
    
def read_data1():
    f = open("Dataset_1.txt","r")
    data = [line.split(" ")[0:2] for line in f]
    data = np.asarray(data)
    data = np.array(data).astype(np.float)
    return data

def read_data2():
    f = open("Dataset_2.txt","r")
    data = [line.split("  ")[1:4] for line in f]
    data = np.asarray(data)
    data = np.array(data).astype(np.float)
    return data

def plotGraph(costs,n,it,d):
 
    x = [i for i in range(0,n)]  
    y = costs

    plt.plot(x, y, label = "Training") 
    
    plt.legend()
      
    plt.xlabel('Iterations') 
    plt.ylabel('Cost') 
      
    plt.title('SSE vs iterations')
    
    filename = "Kmeans_" + d + "_" + str(it) + ".jpg"
    plt.savefig(filename)
    plt.show()
    plt.gcf().clear()


if __name__ == "__main__":
    nsse = []
    for it in range(5):
        data = read_data2()
        
        k = 3 # No.of clusters
        kms = kmeans()
        centres, clusters, sse,n = kms.train(k,data)
        print(sse)
        nsse.append(sse[-1])
        centres = np.array(centres)
        plotGraph(sse,n,it,"d2")
        colors = ['y', 'c', 'b', 'r', 'g', 'm']
        fig, ax = plt.subplots()
        for i in range(k):
            points = np.array([data[j] for j in range(len(clusters)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(centres[:, 0], centres[:, 1], marker='*', s=300, c='#050505')
        filename = "Kmeans_d2c_" + str(it) + ".jpg"
        plt.savefig(filename)
        plt.gcf().clear()
    nsse = np.array(nsse)
    print("Lowest SSE for dataset 2 at r = ", np.argmin(nsse))

    nsse = []
    for it in range(5):
        data = read_data1()
       
        k = 2 # No.of clusters
        kms = kmeans()
        centres, clusters, sse,n = kms.train(k,data)
        print(sse)
        centres = np.array(centres)
        nsse.append(sse[-1])
        plotGraph(sse,n,it,"d1")
        colors = ['y', 'c', 'b', 'r', 'g', 'm']
        fig, ax = plt.subplots()
        for i in range(k):
            points = np.array([data[j] for j in range(len(clusters)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(centres[:, 0], centres[:, 1], marker='*', s=300, c='#050505')
        filename = "Kmeans_d1c_" + str(it) + ".jpg"
        plt.savefig(filename)
        plt.gcf().clear()
        
    nsse = np.array(nsse)
    print("Lowest SSE for dataset 1 at r = ", np.argmin(nsse))