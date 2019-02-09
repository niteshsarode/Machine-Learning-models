import math
import random
import numpy as np

states = 5 # S1, S2, S3, inital and final
symbols = 5 # A, B, C, D, null
dictSymbols = {"A": 0, "B": 1, "C": 2, "D": 3, "O": 4} # consider O as null symbol
stringSet1 = ["AABBCCDD","ABBCBBDD","ACBCBCD","AD","ACBCBABCDD","BABAADDD","BABCDCC","ABDBBCCDD","ABAAACDCCD","ABD"]
stringSet2 = ["DDCCBBAA","DDABCBA","CDCDCBABA","DDBBA","DADACBBAA","CDDCCBA","BDDBCAAAA","BBABBDDDCD","DDADDBCAA","DDCAAA"]
testStrings = ["ABBBCDDD","DADBCBAA","CDCBABA","ADBBBCD"]
string3 = "BADBDCBA"

# Initialize transmission and emission matrix with random numbers from 0 to 1.
def getRandomAandB():
	# Randomizing matrices A and B
	random.seed(99)
	A = [[0] + [random.random() for i in range(3)] + [0]]
	B = [[0 for i in range(symbols)]]
	for i in range(3):
	    transition = [0]+[random.random() for i in range(4)]
	    emission = [random.random() for i in range(symbols)]
	    A.append(transition)
	    B.append(emission)
	A.append([0 for i in range(states-1)] + [1])
	B.append([0 for i in range(symbols)])
	B[0][-1] = 1
	B[-1][-1] = 1
	A = np.array(A)
	B = np.array(B)
	return A,B

#Normalize the random generated numbers so that the sum of numbers along the row should equal to 1.
def normalizeAandB(A,B):
	# Normalizing the matrices
	for i in range(states):
	    total = sum(A[i])
	    A[i] = A[i]/total	
	    total = sum(B[i])
	    if total != 0:
        	B[i] = B[i]/total
	return A,B

# Calculate alpha for Forward Probabilities
def calcAlpha(A,B,string):
    length = len(string)
    alpha = np.zeros((length,states))
    alpha[0,:] = [A[0][j]*B[j][dictSymbols[string[0]]] for j in range(states)]
    for j in range(1,length):
        for i in range(states):
            sumA = 0
            for k in range(states):
                sumA += alpha[j-1][k] * A[k][i]
            alpha[j][i] = sumA*B[i][dictSymbols[string[j]]]
    return alpha

# Calculate beta for Backward Probabilities
def calcBeta(A,B,string):
    length = len(string)
    beta = np.zeros((length,states))
    beta[length-1,:] = [1 for i in range(states)]

    for t in range(length-2,-1,-1):
        for i in range(states):
            s = 0
            for j in range(states):
                s += B[j][dictSymbols[string[t+1]]]*beta[t+1,j] * A[i][j]
            beta[t,i] = s
    return beta

#Expectation Maximization
def calcXi(t,i,j,string,alpha,beta,A,B):
    n = alpha[t,i] * A[i][j] * B[j][dictSymbols[string[t+1]]] * beta[t+1][j]
    d = 0
    for ti in range(states):
        for tj in range(states):
        	d += alpha[t,ti] * A[ti][tj] * B[tj][dictSymbols[string[t+1]]] * beta[t+1][tj]	
    return n/d

def calcGamma(t,i,string,alpha,beta,A,B):
    g = 0
    for j in range(states):
        g += calcXi(t,i,j,string,alpha,beta,A,B) # γt(it) = summation (ξt(it, it+1))
    return g

def calcPi(string,alpha,beta,A,B):
    pi = []
    for i in range(states):
        pi.append(calcGamma(0,i,string,alpha,beta,A,B))
    return pi

def calcA(string,alpha,beta,A,B):
	T = len(string)
	nA = np.zeros((states, states))
	for i in range(states):
	    for j in range(states):
	        n = 0
	        d = 0          
	        for t in range(T-1):
	            n += calcXi(t,i,j,string,alpha,beta,A,B)
	            d += calcGamma(t,i,string,alpha,beta,A,B)
	        if n:
	            nA[i][j] = n/d
	        else:
	            nA[i][j] = A[i][j]
	nA[-1,-1] = 1
	return nA

def calcB(string,alpha,beta,A,B):
    T = len(string)
    nB = np.zeros((states,symbols))
    for j in range(states):
        for k in range(symbols):
            n = 0
            d = 0
            for t in range(T-1):
                if dictSymbols[string[t]] == k:
                	n += calcGamma(t,j,string,alpha,beta,A,B)
                d += calcGamma(t,j,string,alpha,beta,A,B)
            if n:
                nB[j][k] = n/d
            else:
                nB[j][k] = B[j][k]
    nB[0][dictSymbols['O']] = 1
    nB[-1][dictSymbols['O']] = 1
    return nB

def calcProb(string,A,B):
	alpha = calcAlpha(A,B,string)
	length = len(string)
	sumP = 0
	for i in range(states):
	    sumP += alpha[length-1][i]
	return sumP

def calcPrior(A1,B1,A2,B2):
	prob1 = calcProb(string3,A1,B1)
	prob2 = calcProb(string3,A2,B2)
	print(prob1,prob2)
	r = prob1/prob2
	print(r)
	prior1 = 1/(1+r)
	prior2 = 1 - prior1
	print(prior1,prior2)


def trainModel(stringSet):
	A,B = getRandomAandB()
	A,B = normalizeAandB(A,B)
	
	# Training model
	for string in stringSet:
		alpha = calcAlpha(A,B,string)
		beta = calcBeta(A,B,string)
		A = calcA(string,alpha,beta,A,B)
		B = calcB(string,alpha,beta,A,B)
		A[0] = calcPi(string,alpha,beta,A,B)
		print("Transition Matrix after string:",string,"\n",A)
		print("Emission Matrix after string:",string,"\n",B)
	return A,B

def main():
	A1,B1 = trainModel(stringSet1)
	print("Probabilities for given strings in model 1:\n")
	probs = []
	for string in testStrings:
		probs.append({string: calcProb(string,A1,B1)})
	print (probs)


	A2,B2 = trainModel(stringSet2)
	print("\nProbabilities for given strings in model 2:\n")
	probs = []
	for string in testStrings:
		probs.append({string: calcProb(string,A2,B2)})
	print (probs)

	calcPrior(A1,B1,A2,B2)
	


if __name__ == "__main__":
	main()