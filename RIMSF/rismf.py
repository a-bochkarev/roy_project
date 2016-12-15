import numpy as np

def RISMF(S,M,sizeM,learningRate = 0.1, regularizedFactor = 0.1 , K = 1) :
    #separate the sampling operator into training (W1) and validation (W2) set
    W1 = S[0:9*len(S)/10,:]
    W2 = S[9*len(S)/10:len(S),:]
    #initialize randomly P and Q in interval [-0.01,0.01]
    P = np.random.rand(sizeM[0], K)/50 - 0.01
    Q = np.random.rand(K,sizeM[1])/50 - 0.01
    Popt = np.zeros(shape=P.shape)
    Qopt = np.zeros(shape=Q.shape)
    #initialize the minimum RMSE
    minRMSE = float('inf')
    gradEp = np.zeros(K)
    gradEq = np.zeros(K)
    #initialize variables for stopping algorithm
    notDecreaseRMSE = 0
    cond = True
    while(cond) :
        k=0
        for(u,i) in W1 :
            e = M[k]-np.dot(P[u,:],Q[:,i])
            ePrime = (e**2 + regularizedFactor*P[u,:]*P[u,:].T + regularizedFactor*Q[:,i].T*Q[:,i])/2
            gradEp = -e*Q[:,i]+regularizedFactor*P[u,:]
            gradEq = -e*P[u,:]+regularizedFactor*Q[:,i]
            P[u,:] = P[u,:] - learningRate*gradEp
            Q[:,i] = Q[:,i] - learningRate*gradEq
            k=k+1
        #compare RMSE of the validation set W2
        currentRMSE = RMSE(M,W1,W2,P,Q)
        if(currentRMSE<minRMSE) :
            minRMSE = currentRMSE
            Popt = P
            Qopt = Q
            notDecreaseRMSE = 0
        else :
            notDecreaseRMSE = notDecreaseRMSE + 1
        cond = notDecreaseRMSE!=2
    return  np.dot(Popt,Qopt)

def RMSE(M,W1,W2,P,Q) :
    sse = 0
    size = len(W1)
    for (i,j) in W2 :
        sumTemp = 0
        for k in range(P.shape[1]) :
            sumTemp = sumTemp + P[i,k]*Q[k,j]
            sse = sse + (M[size]-sumTemp)**2
        size = size + 1
    return np.sqrt(sse/size)
