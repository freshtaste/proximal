import numpy as np
from dgp import DGP
from est import EST
import matplotlib.pyplot as plt


def simulation(ntrails=2000, T0=1000, T=2000, F=2, K=2, xi=0, static=False, ar_err=False, lag=0):
    mse_tau, mse_alpha, cover_tau = np.zeros((ntrails, 6)), np.zeros((ntrails, 6)), np.zeros((ntrails,6))
    for i in range(ntrails):
        data = DGP(T0, T, F, K, xi, static, ar_err)
        data.get_data()
        if xi==0:
            cy, cw, cx = None, None, None
        else:
            cy, cw, cx = data.cY, data.cW, data.cX
        
        alpha0 = data.beta
        tau0 = 1
        est = EST(data.Y, data.W0, data.X0, data.W1, data.X1, data.T0, cy, cw, cx, lag)
        
        tau, taut = est.sc()
        mse_tau[i,0] = (tau - tau0)**2
        
        tau, taut, alpha, se_tau = est.ols()
        mse_tau[i,1] = (tau - tau0)**2
        mse_alpha[i,1] = np.sqrt(np.sum((alpha - alpha0)**2))
        cover_tau[i,1] = 1 if tau0 <= tau + 1.96*se_tau and tau0>= tau - 1.96*se_tau else 0
        
        tau, taut, alpha, se_tau = est.ols_surrogate()
        mse_tau[i,2] = (tau - tau0)**2
        mse_alpha[i,2] = np.sqrt(np.sum((alpha - alpha0)**2))
        cover_tau[i,2] = 1 if tau0 <= tau + 1.96*se_tau and tau0>= tau - 1.96*se_tau else 0
        
        tau, taut, alpha, se_tau = est.pi()
        mse_tau[i,3] = (tau - tau0)**2
        mse_alpha[i,3] = np.sqrt(np.sum((alpha - alpha0)**2))
        cover_tau[i,3] = 1 if tau0 <= tau + 1.96*se_tau and tau0>= tau - 1.96*se_tau else 0
        
        tau, taut, alpha, se_tau = est.pi_surrogate_post()
        mse_tau[i,4] = (tau - tau0)**2
        mse_alpha[i,4] = np.sqrt(np.sum((alpha - alpha0)**2))
        cover_tau[i,4] = 1 if tau0 <= tau + 1.96*se_tau and tau0>= tau - 1.96*se_tau else 0
        
        tau, taut, alpha, se_tau = est.pi_surrogate()
        mse_tau[i,5] = (tau - tau0)**2
        mse_alpha[i,5] = np.sqrt(np.sum((alpha - alpha0)**2))
        cover_tau[i,5] = 1 if tau0 <= tau + 1.96*se_tau and tau0>= tau - 1.96*se_tau else 0
    
    return np.mean(mse_tau, axis=0), np.mean(mse_alpha, axis=0), np.mean(cover_tau, axis=0)


mse_tau, mse_alpha, cover_tau = simulation(ntrails=2000, T0=100, T=200, F=1, K=1, xi=0, static=False, ar_err=True)
#print(mse_tau)
#print(mse_alpha)
#print(cover_tau)

if __name__ == '__main__':
    Ks = [1, 5]
    Ts = [200, 800]
    output = np.zeros((len(Ks)*len(Ts)*2*2+1,5*2))

    mse = np.zeros((len(Ks)*len(Ts),5*2))
    cover = np.zeros((len(Ks)*len(Ts),5*2))
    i = 0
    for k in Ks:
        for t in Ts:
            print(k,t)
            mse_tau, mse_alpha, cover_tau = simulation(ntrails=2000, T0=int(t/2), T=t, F=k, K=k, xi=0, static=True, ar_err=False)
            mse[i,:5] = mse_tau[1:]/mse_tau[-1]
            cover[i,:5] = cover_tau[1:]
            mse_tau, mse_alpha, cover_tau = simulation(ntrails=2000, T0=int(t/2), T=t, F=k, K=k, xi=0, static=False, ar_err=False)
            mse[i,5:] = mse_tau[1:]/mse_tau[-1]
            cover[i,5:] = cover_tau[1:]
            i+=1
    output[:len(mse),:] = mse
    output[len(mse):2*len(mse),:] = cover

    mse = np.zeros((len(Ks)*len(Ts),5*2))
    cover = np.zeros((len(Ks)*len(Ts),5*2))
    i = 0
    for k in Ks:
        for t in Ts:
            print(k,t)
            mse_tau, mse_alpha, cover_tau = simulation(ntrails=2000, T0=int(t/2), T=t, F=k, K=k, xi=0, static=True, ar_err=True, lag=1)
            mse[i,:5] = mse_tau[1:]/mse_tau[-1]
            cover[i,:5] = cover_tau[1:]
            mse_tau, mse_alpha, cover_tau = simulation(ntrails=2000, T0=int(t/2), T=t, F=k, K=k, xi=0, static=False, ar_err=True, lag=1)
            mse[i,5:] = mse_tau[1:]/mse_tau[-1]
            cover[i,5:] = cover_tau[1:]
            i+=1
            
    output[2*len(mse)+1:3*len(mse)+1,:] = mse
    output[3*len(mse)+1:,:] = cover
    np.savetxt('simluation_results.csv', output, delimiter=',')

