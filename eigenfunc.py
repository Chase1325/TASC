import numpy as np
import GPy
from matplotlib import pyplot as plt
import time

def func(x):
    return -(1.4-3*x)*np.sin(18*x)
    """Latent function."""
    #return 1.0 * np.sin(x * 3 * np.pi) + \
     #      0.3 * np.cos(x * 9 * np.pi) + \
      #     0.5 * np.sin(x * 7 * np.pi)
np.random.seed(52)

N = 50
noise_var = .01

X_a = (np.random.rand(N)*0.6)[:,None]#np.linspace(0,0.6,N)[:,None]
X_b = (np.random.rand(N)*0.6 + 0.6)[:,None]#np.linspace(0.6,1.2,N)[:,None]
k = GPy.kern.RBF(1)
#print(np.random.normal(size=(N, 1)))
y_a = func(X_a) + np.sqrt(noise_var) * np.random.normal(size=(N, 1))# np.random.multivariate_normal(np.zeros(N),k.K(X_a)+np.eye(N)*np.sqrt(noise_var)).reshape(-1,1)
y_b = func(X_a) + np.sqrt(noise_var) * np.random.normal(size=(N, 1))#np.random.multivariate_normal(np.zeros(N),k.K(X_b)+np.eye(N)*np.sqrt(noise_var)).reshape(-1,1)
X = np.append(X_a, X_b, axis=0)#np.linspace(0,100,1000)[:,None]
y = np.append(y_a, y_b, axis=0)#np.random.multivariate_normal(np.zeros(N*2),k.K(X)+np.eye(N*2)*np.sqrt(noise_var)).reshape(-1,1)

print(X_a.shape, X_b.shape, X.shape)

start = time.time()

hyperGrid = dict()
hyperGrid[0] = GPy.models.GPRegression(X[:20],y[:20], noise_var=noise_var)
hyperGrid[0].optimize('lbfgs')
for i in range(9):
    n = i+1
    hyperGrid[n] = GPy.models.GPRegression(X[i*10:i*10+30],y[i*10:i*10+30], noise_var=noise_var)
    hyperGrid[n].optimize('lbfgs')
    
print(time.time()-start)
m_a = GPy.models.GPRegression(X_a[:30],y_a[:30], noise_var=noise_var)
m_a.optimize('lbfgs')
m_b = GPy.models.GPRegression(X_b[:10],y_b[:10], noise_var=noise_var)
m_b.optimize('lbfgs')

start = time.time()
m_full = GPy.models.GPRegression(X,y, noise_var=noise_var)
m_full.optimize('lbfgs')
print(time.time()-start)

pt = np.array([[60]])
mu_a, v_a = m_a.predict(pt)
mu_b, v_b = m_b.predict(pt)
print(mu_a, v_a, mu_b, v_b)

mu_est = (v_b*mu_a + v_a*mu_b)/(v_a+v_b)
v_est = (v_a*v_b)/(v_a+v_b)

mu_f, v_f = m_full.predict(pt)

print('true', func(60))
print(mu_est, v_est, mu_f, v_f)
keys = hyperGrid.keys()
print(keys)
val = 0
for key in keys:
    val += hyperGrid[key].log_likelihood()
    hyperGrid[key].plot()
print(val)
print(m_a.log_likelihood(), m_b.log_likelihood(),m_a.log_likelihood()+m_b.log_likelihood(), m_full.log_likelihood())
#m_a.plot()
#m_b.plot()
m_full.plot()
print(m_full)
plt.show()



#print m_full

#Z = np.random.rand(100,1)*100
#start = time.time()
#m = GPy.models.SparseGPRegression(X,y,Z=Z)
#m.optimize('bfgs')
#print(time.time()-start)
#m.plot()
#plt.show()
#print(m['inducing_inputs'], Z)
#print(m.log_likelihood(), m_full.log_likelihood())