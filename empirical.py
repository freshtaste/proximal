import numpy as np
from scipy.optimize import fmin_slsqp
from toolz import partial
import statsmodels.api as sm
from utils import hac
from est import EST

class EST_extended(EST):
    
    def __init__(self, Y, W, X, Z0, Z1, T0, t1, Cy=None, Cw=None, Cx=None, lag=0):
        super().__init__(Y, W, X, Z0, Z1, T0, Cy, Cw, Cx, lag)
        self.t1 = t1
        self.X_original = X.copy()
        self.clean_surrogates2()
        
    def ols(self):
        X1 = np.zeros(self.T)
        X1[self.T0:self.T0+self.t1] += 1
        if self.Cy is not None:
            regressor = np.column_stack((X1, self.W, self.Cy))
        else:
            regressor = np.column_stack((X1, self.W))
        regressor = sm.add_constant(regressor)
        model = sm.OLS(self.Y, regressor).fit(cov_type='hac',cov_kwds={'maxlags': self.lag})
        tau = model.params[1]
        alpha = model.params[2:2+self.W.shape[1]]
        taut = self.Y - self.W.dot(alpha) - model.params[0]
        se_tau = model.bse[1]
        return tau, taut, alpha, se_tau
    
    def ols_surrogate(self):
        X1 = np.copy(self.X)
        X1 = sm.add_constant(X1)
        X1[:self.T0] = 0
        if self.Cy is not None:
            regressor = np.column_stack((X1, self.W, self.Cy))
        else:
            regressor = np.column_stack((X1, self.W))
        regressor = sm.add_constant(regressor)
        model = sm.OLS(self.Y, regressor).fit()
        gamma = model.params[1:1+X1.shape[1]]
        alpha = model.params[1+X1.shape[1]:X1.shape[1]+self.W.shape[1]+1]
        taut = X1.dot(gamma)
        taut = (self.Y - self.W.dot(alpha) - model.params[0])
        #taut = self.Y - self.W.dot(alpha)
        tau = np.mean(taut[self.T0:self.T0+self.t1])
        # inference with GMM
        U0 = regressor.T * (self.Y - regressor.dot(model.params))
        U1 = X1.dot(gamma) - tau
        U1[:self.T0] *= 0
        U = np.column_stack((U0.T, U1))
        G = np.zeros((U.shape[1], U.shape[1]))
        G[:regressor.shape[1],:regressor.shape[1]] = regressor.T @ regressor/self.T
        G[-1,-1] = self.t1/self.T
        G[-1,1:1+len(gamma)] = -np.sum(X1[self.T0:self.T0+self.t1],axis=0)/self.T
        Omega = hac(U, self.lag)
        Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G).T
        var_tau = Cov[-1,-1]
        se_tau = np.sqrt(var_tau/self.T)
        return tau, taut, alpha, se_tau
    
    def pi(self):
        if self.W.shape[1] == self.Z0.shape[1]:
            if self.Cw is not None and self.Cy is not None:
                Z0 = np.column_stack((self.Z0, self.Cy, self.Cw))
                W = np.column_stack((self.W, self.Cy, self.Cw))
            else:
                Z0 = self.Z0
                W = self.W
            Y = self.Y
            Z0W = Z0[:self.T0].T @ W[:self.T0]
            Z0Y = Z0[:self.T0].T @ Y[:self.T0]
            alpha = np.linalg.solve(Z0W, Z0Y)
            taut = Y - W.dot(alpha)
            tau = np.mean(taut[self.T0:self.T0+self.t1])
            # inference with GMM
            U0 = Z0.T * (self.Y - W.dot(alpha)) 
            U1 = Y - tau - W.dot(alpha)
            U0[:,self.T0:] *= 0
            U1[:self.T0] *= 0
            U = np.column_stack((U0.T, U1))
            G = np.zeros((U.shape[1], U.shape[1]))
            dimZ0, dimW = Z0.shape[1], W.shape[1]
            G[:dimZ0,:dimW] = Z0W/self.T
            G[-1,:dimW] = np.sum(W[self.T0:self.T0+self.t1],axis=0)/self.T
            G[-1,-1] = self.t1/self.T
            Omega = hac(U, self.lag)
            Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G.T)
            var_tau = Cov[-1,-1]
            se_tau = np.sqrt(var_tau/self.T)
        else:
            RuntimeError("Not implemented yet.")
        return tau, taut, alpha[:self.W.shape[1]], se_tau
    
    def pi_surrogate(self):
        if self.W.shape[1] == self.Z0.shape[1] and self.X.shape[1] == self.Z1.shape[1]:
            if self.Cw is not None and self.Cy is not None and self.Cx is not None:
                Z0 = np.column_stack((self.Z0, self.Cy, self.Cw))
                W = np.column_stack((self.W, self.Cy, self.Cw))
                Z1 = np.column_stack((self.Z1, self.Cx))
                X = np.column_stack((self.X, self.Cx))
            else:
                Z0 = self.Z0
                W = self.W
                Z1 = self.Z1
                X = self.X
            Y = self.Y
            Z0W = Z0[:self.T0].T @ W[:self.T0]
            Z0Y = Z0[:self.T0].T @ Y[:self.T0]
            alpha = np.linalg.solve(Z0W, Z0Y)
            tauhat = Y[self.T0:] - W[self.T0:].dot(alpha)
            Z1X = Z1[self.T0:].T @ X[self.T0:]
            Z1tau = Z1[self.T0:].T @ tauhat
            gamma = np.linalg.solve(Z1X, Z1tau)
            taut = X.dot(gamma)
            taut[:self.T0] = (Y - W.dot(alpha))[:self.T0] 
            #taut = (Y - W.dot(alpha))
            tau = np.mean(taut[self.T0:self.T0+self.t1])
            #print("diff: ",np.mean(tauhat), tau, np.mean(tauhat-X.dot(gamma)[self.T0:]))
            # inference with GMM
            U0 = Z0.T * (Y - W.dot(alpha)) 
            U1 = Z1.T * (Y - W.dot(alpha) - X.dot(gamma))
            U2 = X.dot(gamma) - tau
            U0[:,self.T0:] *= 0
            U1[:,:self.T0] *= 0
            U2[:self.T0] *= 0
            U = np.column_stack((U0.T, U1.T, U2))
            G = np.zeros((U.shape[1], U.shape[1]))
            dimZ0, dimZ1, dimW, dimX = Z0.shape[1], Z1.shape[1], W.shape[1], X.shape[1]
            G[:dimZ0,:dimW] = Z0W/self.T
            G[dimZ0:dimZ0+dimZ1,:dimW] = Z1[self.T0:].T @ W[self.T0:]/self.T
            G[dimZ0:dimZ0+dimZ1,dimW:dimW+dimX] = Z1[self.T0:].T @ X[self.T0:]/self.T
            G[-1,dimW:dimW+dimX] = -np.sum(X[self.T0:self.T0+self.t1],axis=0)/self.T
            G[-1,-1] = self.t1/self.T
            Omega = hac(U, self.lag)
            Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G).T
            var_tau = Cov[-1,-1]
            se_tau = np.sqrt(var_tau/self.T)
            #print("z gamma: ", gamma/np.sqrt(np.diag(Cov)[dimW:dimW+dimX]/self.T))
        else:
            RuntimeError("Not implemented yet.")
        return tau, taut, alpha[:self.W.shape[1]], se_tau
        
    def clean_surrogates(self):
        T = self.T
        T0 = self.T0
        X = np.copy(self.X_original)
        tauts = np.zeros_like(X)
        for i in range(X.shape[1]):
            X1 = X[:, i]
            X1 = sm.add_constant(X1)
            X1[:T0] = 0
            if self.Cy is not None:
                regressor = np.column_stack((X1, self.W, self.Cy))
            else:
                regressor = np.column_stack((X1, self.W))
            regressor = sm.add_constant(regressor)
            model = sm.OLS(self.Y, regressor).fit(cov_type='hac',cov_kwds={'maxlags': self.lag})
            alpha = model.params[2:2+self.W.shape[1]]
            taut = self.Y - self.W.dot(alpha) - model.params[0]
            tauts[:, i] = taut
        self.tauts = tauts
        self.X = tauts

    def clean_surrogates2(self):
        tauts = []
        for i in range(self.X.shape[1]):
            X1 = np.copy(self.X[:, i])
            if self.Cy is not None:
                Z0 = np.column_stack((self.Z0, self.Cy))
                W = np.column_stack((self.W, self.Cy))
            else:
                Z0 = self.Z0
                W = self.W
            Y = X1
            Z0W = Z0[:self.T0].T @ W[:self.T0]
            Z0Y = Z0[:self.T0].T @ Y[:self.T0]
            alpha = np.linalg.solve(Z0W, Z0Y)
            taut = Y - W.dot(alpha)
            tauts.append(taut)
        self.X = np.column_stack(tauts)
        return self.X

    
    
def rolling(Y, W, X, Z0, Z1, T0, Cy=None, Cw=None, Cx=None, lag=0):
    T = len(Y)
    tau_list = []
    se_list = []
    for t in range(50,T-T0):
        est = EST_extended(Y, W, X, Z0, Z1, T0, t, Cy, Cw, Cx, lag)
        tau_ols, _, _, se_tau_ols = est.ols()
        tau_ols_sur, _, _, se_tau_ols_sur = est.ols_surrogate()
        tau_pi, _, _, se_tau_pi = est.pi()
        tau_pi_sur, _, _, se_tau_pi_sur = est.pi_surrogate()
        tau_pi_sur_post, _, _, se_tau_pi_sur_post = est.pi_surrogate_post()
        tau_list.append([tau_ols, tau_ols_sur, tau_pi, tau_pi_sur, tau_pi_sur_post])
        se_list.append([se_tau_ols, se_tau_ols_sur, se_tau_pi, se_tau_pi_sur, se_tau_pi_sur_post])
    Tau = np.array(tau_list)
    Se = np.array(se_list)
    Lower = Tau - 1.96*Se
    Upper = Tau + 1.96*Se
    return Tau, Se, Lower, Upper


import matplotlib.pyplot as plt

def plot_time_series(Tau, Lower, Upper):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # First plot
    axs[0].plot(Tau[:,0], label='OLS')
    axs[0].fill_between(range(len(Tau)), Lower[:,0], Upper[:,0], alpha=0.3)
    axs[0].plot(Tau[:,1], label='OLS Surrogate')
    axs[0].fill_between(range(len(Tau)), Lower[:,1], Upper[:,1], alpha=0.3)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Tau')
    axs[0].set_title('OLS vs OLS Surrogate')
    axs[0].legend()
    
    # Second plot
    axs[1].plot(Tau[:,2], label='PI')
    axs[1].fill_between(range(len(Tau)), Lower[:,2], Upper[:,2], alpha=0.3)
    axs[1].plot(Tau[:,3], label='PI Surrogate')
    axs[1].fill_between(range(len(Tau)), Lower[:,3], Upper[:,3], alpha=0.3)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Tau_t')
    axs[1].set_title('PI vs PI Surrogate')
    axs[1].legend()
    
    # Third plot
    #axs[2].plot(Tau[:,4], label='PI Surrogate Post')
    #axs[2].fill_between(range(len(Tau)), Lower[:,4], Upper[:,4], alpha=0.3)
    #axs[2].set_xlabel('Time')
    #axs[2].set_ylabel('Tau_t')
    #axs[2].set_title('PI Surrogate Post')
    #axs[2].legend()
    
    plt.show()
