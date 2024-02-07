import numpy as np

class DGP(object):
    
    def __init__(self, T0, T, F, K, xi, static=False, ar_err=False):
        self.T0 = T0
        self.T = T
        self.F = F
        self.K = K
        self.xi = xi
        self.theta = np.ones(K)
        self.beta = np.ones(F)
        self.static = static
        self.ar_err = ar_err

    def get_eps(self):
        if self.ar_err:
            nuY = np.random.normal(size=(self.T,))
            nuW = np.random.normal(size=(self.T,self.F*2))
            nuX = np.random.normal(size=(self.T,self.K*2))
            nuD = np.random.normal(size=(self.T,))
            self.epsY, self.epsW, self.epsX, self.delta = np.zeros(self.T), np.zeros((self.T,self.F*2)), np.zeros((self.T,self.K*2)), np.zeros(self.T)
            phi = 0.1
            for t in range(1, self.T):
                self.epsY[t] = self.epsY[t-1]*phi + nuY[t]
                self.epsW[t,:] = self.epsW[t-1,:]*phi + nuW[t,:]
                self.epsX[t,:] = self.epsX[t-1,:]*phi + nuX[t,:]
                self.delta[t] = self.delta[t-1]*phi + nuD[t]
        else:
            self.epsY = np.random.normal(size=(self.T,))
            self.epsW = np.random.normal(size=(self.T,self.F*2))
            self.epsX = np.random.normal(size=(self.T,self.K*2))
            self.delta = np.random.normal(size=(self.T,))
        
    def get_factors(self):
        self.lam = (np.log(np.linspace(1,self.T,self.T)) + np.random.normal(size=(self.T,self.F)).T).T
        if self.static:
            self.lam = np.random.normal(size=(self.T,self.K)) + 1
        self.rho = np.random.normal(size=(self.T,self.K))
        self.rho[:,0] += 1
        
    def get_covariates(self):
        self.cY = np.random.normal(size=(self.T,))
        self.cW = np.random.normal(size=(self.T,self.F*2))
        self.cX = np.random.normal(size=(self.T,self.K*2))
        
    def get_vars(self):
        self.Y = self.lam.dot(self.beta) + self.cY*self.xi + self.epsY
        self.taut = self.rho[self.T0:].dot(self.theta)
        self.tau = np.mean(self.taut)
        self.Y[self.T0:] += self.taut + self.delta[self.T0:]
        self.W0 = self.lam + self.cW[:,:self.F]*self.xi + self.epsW[:,:self.F]
        self.W1 = self.lam + self.cW[:,self.F:]*self.xi + self.epsW[:,self.F:]
        self.X0 = self.rho + self.cX[:,:self.K]*self.xi + self.epsX[:,:self.K]
        self.X1 = self.rho + self.cX[:,self.K:]*self.xi + self.epsX[:,self.K:]
        
    def get_data(self):
        self.get_eps()
        self.get_factors()
        self.get_covariates()
        self.get_vars()
    
    
if __name__ == "__main__":
    dgp = DGP(200, 400, 1, 1, 0, False)
    dgp.get_data()
    print(dgp.Y)