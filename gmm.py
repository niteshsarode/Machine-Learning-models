from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import kmeans

class GMM(object):
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.N = X.shape[0]
        self.logl = []
        self.resp = None
        self.params = self.initialize()
        self.likelihood = 0.0

    def initialize(self):
        params = {}

        coeffs = np.random.random(size=(self.K,))
        coeffs /= np.sum(coeffs)

        min_data = np.min(self.X, axis=0)
        max_data = np.max(self.X, axis=0)

        for i in range(1, self.K + 1):
            params['means{}'.format(i)] = np.random.uniform(min_data, max_data)
            params['covs{}'.format(i)] = np.cov(self.X.T)
            params['coeff{}'.format(i)] = coeffs[i-1]
        return params

    def gauss(self, x, mean, cov):
        return stats.multivariate_normal.pdf(x, mean=mean, cov=cov)
    
    def getclusters(self, resp):
        clusters = []
        for i in range(self.N):
            clusters.append(np.argmax(resp[i]))
        return clusters
    
    def expectation(self):
        E = np.zeros((self.N,self.K))
        for k in range(1,self.K+1):
            for n in range(self.N):
                x_n = self.X[n,:]
                sumMix = 0.0
                for j in range(1,self.K+1):
                    pi_j = self.params['coeff{}'.format(j)]
                    mean_j = self.params['means{}'.format(j)]
                    covars_j = self.params['covs{}'.format(j)]
                    gauss_nj = self.gauss(x_n, mean_j, covars_j)
                    if j == k:
                        numerator = gauss_nj * pi_j
                    sumMix += gauss_nj * pi_j
                gamma_nk = numerator / float(sumMix)
                E[n,k-1] = gamma_nk
        return E

    def maximization(self):
        A = {}
        for k in range(1,self.K+1):
            N_k = sum(self.resp[:,k-1])
            mean_k = (1 / N_k) * np.dot(self.X.T,self.resp[:,k-1])

            cov_sum = 0.0
            for n in range(self.N):
                x_n = self.X[n,:]
                diff = x_n - mean_k
                cov_sum += self.resp[n,k-1] * (np.outer(diff,diff))
            covars_k = cov_sum / N_k

            pi_k = N_k / self.N

            A['means{}'.format(k)] = mean_k
            A['covs{}'.format(k)] = covars_k
            A['coeff{}'.format(k)] = pi_k
        return A
    
    def loglikelihood(self):
        logl = 0.0
        for n in range(self.N):
            x_n = self.X[n,:]
            l = 0.0
            for j in range(1,self.K+1):
                pi_j = self.params['coeff{}'.format(j)]
                mean_j = self.params['means{}'.format(j)]
                covars_j = self.params['covs{}'.format(j)]
                gauss_nj = self.gauss(x_n, mean_j, covars_j)
                l += gauss_nj * pi_j
            logl += np.log(l)
        return logl/self.N
                

    def train(self, n_it):
        for i  in range(n_it):
            print("Iteration: ",i)
            self.resp = self.expectation()
            self.params = self.maximization()
            ll = round(self.loglikelihood(),6)
            if self.likelihood == ll:
                break
            self.likelihood = ll
            self.logl.append(self.likelihood)
        clusters = self.getclusters(self.resp)
        return self.params, clusters, self.logl
    
    def plotGraph(self,logl,n,d):
        x = [i for i in range(0,len(logl))]  
        y = logl
        plt.plot(x, y, label = "Training") 

        plt.legend()

        plt.xlabel('Iterations') 
        plt.ylabel('LogLikelihood') 

        plt.title('LogL vs iterations')
        filename = "GMM_"+d+".jpg"
        plt.savefig(filename)
        plt.show()
        plt.gcf().clear()

    def plotClusters(self,params,clusters,data,k):
        colors = ['b', 'g', 'r', 'y', 'm', 'c']
        fig, ax = plt.subplots()
        for i in range(0,k):
            mean = params['means{}'.format(i+1)]
            points = np.array([data[j] for j in range(len(clusters)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
            ax.scatter(mean[0], mean[1], marker='*', s=300, c='#050505')
        filename = "GMMc_d"+str(k-1)+".jpg"
        plt.savefig(filename)
        plt.gcf().clear()

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

if __name__ == "__main__":

    data1 = read_data1()
    k = 2
    gmm1 = GMM(data1,k)
    params, clusters, logl = gmm1.train(50)
    gmm1.plotGraph(logl,50,"d1")
    gmm1.plotClusters(params,clusters,data1,k)

    data2 = read_data2()
    k = 3
    gmm2 = GMM(data2,k)
    params, clusters, logl = gmm2.train(50)
    gmm2.plotGraph(logl,50,"d2")
    gmm2.plotClusters(params,clusters,data2,k)


