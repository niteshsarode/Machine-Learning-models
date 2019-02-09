import numpy as np

class BMM(object):
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.N = X.shape[0]
        self.logl = []
        self.params = self.initialize()

    def initialize(self):
        params = {}

        coeffs = np.random.random(size=(self.K,))
        coeffs /= np.sum(coeffs)

        for i in range(1, self.K + 1):
            params['theta{}'.format(i)] = np.random.uniform(0, 1)
            params['coeff{}'.format(i)] = coeffs[i-1]
        return params

    def binomial(self, x, theta):
        M = len(x)
        s = x.tolist().count(1)
        pb = (theta ** s) * ((1-theta)**(M-s))
        return pb
    
    def em(self):
        pH1 = 0
        pT1 = 0
        pH2 = 0
        pT2 = 0
        for n in range(self.N):
            x_n = self.X[n,:]
            pi_1 = self.params['coeff1']
            pi_2 = self.params['coeff2']
            pi_1 = 0.5
            pi_2 = 0.5
            theta_1 = self.params['theta1']
            theta_2 = self.params['theta2']
            b_1 = self.binomial(x_n, theta_1)
            b_2 = self.binomial(x_n, theta_2)
            pH1 += b_1 / (b_1 + b_2) * pi_1 * x_n.tolist().count(1)
            pT1 += b_1 / (b_1 + b_2) * pi_1 * x_n.tolist().count(0)
            pH2 += b_2 / (b_1 + b_2) * pi_2 * x_n.tolist().count(1)
            pT2 += b_2 / (b_1 + b_2) * pi_2 * x_n.tolist().count(0)
        self.params['theta1'] = round(pH1 / (pH1 + pT1),2)
        self.params['theta2'] = round(pH2 / (pH2 + pT2),2)
        self.params['coeff1'] = (pH1 + pT1) / self.N
        self.params['coeff2'] = (pH2 + pT2) / self.N
        return

    
    def loglikelihood(self):
        logl = 0
        for n in range(self.N):
            x_n = self.X[n,:]
            suml = 0
            for j in range(1,self.K+1):
                pi_j = self.params['coeff{}'.format(j)]
                theta_j = self.params['theta{}'.format(j)]
                binomial = self.binomial(x_n,theta_j)
                suml += binomial * pi_j
            logl += np.log(suml)
        return logl/self.N
                

    def train(self, n_it):
        for i  in range(n_it):
            self.em()
            ll = self.loglikelihood()
            if self.logl and ll == self.logl[-1]:
                break
            self.logl.append(self.loglikelihood())
        return self.params, self.logl


def read_data():
    f = open("flips.txt","r")
    data = []
    for line in f:
        line = line.split(" ")[:-1]
        data.append(line)
    return np.array(data).astype(np.float)

if __name__ == "__main__":

    data = read_data()

    bmm = BMM(data,2)
    params, logl = bmm.train(20)
    print("theta1: ", params['theta1'], "theta2: ", params['theta2'])
    print("Log likelihood: ",logl)