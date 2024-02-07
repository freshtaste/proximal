import numpy as np
from scipy.optimize import fmin_slsqp
from toolz import partial
import numpy as np
import statsmodels.api as sm
from utils import hac

class EST(object):
    
    def __init__(self, Y, W, X, Z0, Z1, T0, Cy=None, Cw=None, Cx=None, lag=0):
        self.Y = Y
        self.W = W
        self.X = X
        self.Z0 = Z0
        self.Z1 = Z1
        self.T0 = T0
        self.Cy = Cy
        self.Cw = Cw
        self.Cx = Cx
        self.T = len(Y)
        self.T1 = self.T - self.T0
        self.lag = lag
    
    def sc(self):
        self.alpha = get_w(self.W, self.Y)
        taut = self.Y - self.W.dot(self.alpha)
        tau = np.mean(taut[self.T0:])
        return tau, taut
    
    def ols(self):
        X1 = np.zeros(self.T)
        X1[self.T0:] += 1
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
        tau = np.mean(taut[self.T0:])
        # inference with GMM
        U0 = regressor.T * (self.Y - regressor.dot(model.params))
        U1 = X1.dot(gamma) - tau
        U1[:self.T0] *= 0
        U = np.column_stack((U0.T, U1))
        G = np.zeros((U.shape[1], U.shape[1]))
        G[:regressor.shape[1],:regressor.shape[1]] = regressor.T @ regressor/self.T
        G[-1,-1] = self.T1/self.T
        G[-1,1:1+len(gamma)] = -np.sum(X1[self.T0:],axis=0)/self.T
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
            tau = np.mean(taut[self.T0:])
            # inference with GMM
            U0 = Z0.T * (self.Y - W.dot(alpha)) 
            U1 = Y - tau - W.dot(alpha)
            U0[:,self.T0:] *= 0
            U1[:self.T0] *= 0
            U = np.column_stack((U0.T, U1))
            G = np.zeros((U.shape[1], U.shape[1]))
            dimZ0, dimW = Z0.shape[1], W.shape[1]
            G[:dimZ0,:dimW] = Z0W/self.T
            G[-1,:dimW] = np.sum(W[self.T0:],axis=0)/self.T
            G[-1,-1] = self.T1/self.T
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
            tau = np.mean(taut[self.T0:])
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
            G[-1,dimW:dimW+dimX] = -np.sum(X[self.T0:],axis=0)/self.T
            G[-1,-1] = self.T1/self.T
            Omega = hac(U, self.lag)
            Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G).T
            var_tau = Cov[-1,-1]
            se_tau = np.sqrt(var_tau/self.T)
        else:
            RuntimeError("Not implemented yet.")
        return tau, taut, alpha[:self.W.shape[1]], se_tau
    
    def pi_surrogate_post(self):
        if self.W.shape[1] == self.Z0.shape[1] and self.X.shape[1] == self.Z1.shape[1]:
            if self.Cw is not None and self.Cy is not None and self.Cx is not None:
                Z = np.column_stack((self.Z0, self.Cy, self.Cw, self.Z1, self.Cx))
                WX = np.column_stack((self.W, self.Cy, self.Cw, self.X, self.Cx))
                X = np.column_stack((self.X, self.Cx))
            else:
                Z = np.column_stack((self.Z0, self.Z1))
                WX = np.column_stack((self.W, self.X))
                X = self.X
            ZWX = Z[self.T0:].T @ WX[self.T0:]
            ZY = Z[self.T0:].T @ self.Y[self.T0:]
            params = np.linalg.solve(ZWX, ZY)
            gamma = params[-X.shape[1]:]
            taut = X.dot(gamma)
            tau = np.mean(taut[self.T0:])
            # inference with GMM
            U0 = (Z.T * (self.Y - WX.dot(params)))[:,self.T0:]
            U1 = X[self.T0:].dot(gamma) - tau
            U = np.column_stack((U0.T, U1))
            G = np.zeros((U.shape[1], U.shape[1]))
            G[:Z.shape[1],:WX.shape[1]] = ZWX/self.T1
            G[-1,-X.shape[1]-1:-1] = -np.sum(X[self.T0:],axis=0)/self.T1
            G[-1,-1] = 1
            Omega = hac(U, self.lag)
            Cov = np.linalg.inv(G) @ Omega @ np.linalg.inv(G).T
            var_tau = Cov[-1,-1]
            se_tau = np.sqrt(var_tau/self.T1)
        else:
            RuntimeError("Not implemented yet.")
        return tau, taut, params[:self.W.shape[1]], se_tau 
    

def loss_w(weight, W, y) -> float:
    return np.sqrt(np.mean((y - W.dot(weight))**2))

def get_w(W, y):
    
    w_start = [1/W.shape[1]]*W.shape[1]

    weights = fmin_slsqp(partial(loss_w, W=W, y=y),
                         np.array(w_start),
                         f_eqcons=lambda x: np.sum(x) - 1,
                         bounds=[(0.0, 1.0)]*len(w_start),
                         disp=False)
    return weights
