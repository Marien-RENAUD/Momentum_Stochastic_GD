
import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as nprandom
import numpy.linalg as nplinalg

N = 2*10**2
d = 10**2
Id = np.eye(d)
# epsilon = 5*10**(-0)*np.arange(0,d-1,2)
epsilon = 5*10**(-1)
mat_a = np.eye(d)*np.arange(1,d+1)
mat_a[0,0]=10**(-6)
for i in range(0,d-1,2):
    # mat_a[i+1,i] = epsilon[int(i/2)]
    mat_a[i+1,i] = epsilon
# vec_a = np.eye(d)[:,0]
features_matrix = np.vstack((Id,mat_a.T))
bias = np.zeros(N)
print(features_matrix) 
# vec_a[1]=epsilon

def f(x):
    return 0.5*np.sum(x**2)/(N) + 0.5*(np.sum(np.dot(mat_a.T,x)**2))/(N)
def grad_f(x):
    return np.dot(Id,x)/N + np.dot(mat_a.T,x)/N
def sto_grad_f(x,U):
    if U<int(N/2):
         return x[U]*Id[:,U]*(U<N)
    else:
        U -= int(N/2)
        return np.dot(mat_a.T,x)[U]*mat_a.T[U]

c_0 = 1 # Scaling x_0
# x_0 = np.ones(d)*c_0
x_0 = nprandom.normal(1,1,d)
# x_0[0]=-epsilon/2

n_iter = 3*10**3 # nb itérations

# Params
H= np.dot(mat_a,mat_a.T)+Id
mu,L = np.min(nplinalg.eig(H)[0])/N,np.max(nplinalg.eig(H)[0])/N

f_snag_100 = np.empty(n_iter)
f_snag_150 = np.empty(n_iter)
f_snag_N = np.empty(n_iter)
f_snag_2N = np.empty(n_iter)

racoga_100 = np.empty(n_iter)
racoga_150 = np.empty(n_iter)
racoga_N = np.empty(n_iter)
racoga_2N = np.empty(n_iter)

def racoga_computation(x,N,features_matrix,bias):
    grad =(np.dot(features_matrix,x)-bias).reshape(N,1)*features_matrix
    corr = np.dot(grad,grad.T)
    racoga = np.sum(np.triu(corr,k=1))/np.sum(np.diag(corr))
    # return N/(1+2*racoga)
    return racoga 

def algo(rho):
        ### Nesterov stochastic ###
    n_ech = 20
    # n_iter = 1*10**1
    x_snag = np.empty((n_iter,d))
    y_snag = np.empty((n_iter,d))
    z_snag = np.empty((n_iter,d))
    f_snag = np.empty(n_iter)

    x_snag[0,:] = x_0
    y_snag[0,:] = x_0
    z_snag[0,:] = x_0
    f_snag[0] = f(x_0)

    ## Params
    step = 1/(L*rho)
    eta = 1/(np.sqrt(mu*L)*rho)
    beta = 1-np.sqrt( (mu/L) )/rho
    alpha = 1/(1+(1/rho)*np.sqrt(mu/L))

    count=0
    moy=np.zeros(n_iter)
    racoga = np.zeros(n_iter)
    racoga[0] = n_ech*racoga_computation(y_snag[0,:],N,features_matrix,bias)
    for j in range(n_ech):
        for i in range(1,n_iter):
            U = nprandom.randint(N) 
            x_snag[i,:] = y_snag[i-1,:] - step*sto_grad_f(y_snag[i-1,:],U)
            z_snag[i,:] = beta*z_snag[i-1,:] + (1-beta)*y_snag[i-1,:] - eta*sto_grad_f(y_snag[i-1,:],U)
            ## alpha_i
            y_snag[i,:] = (1-alpha)*z_snag[i,:] + alpha*x_snag[i,:]
            f_snag[i] = f(x_snag[i,:])
            racoga[i] += racoga_computation(y_snag[i,:],N,features_matrix,bias)
            if U==(N-1):
                count+=1
        moy += f_snag
    moy = moy/n_ech
    racoga /= n_iter
    return moy,racoga

f_snag_100,racoga_100 = algo(100)
f_snag_150,racoga_150 = algo(150)
f_snag_N,racoga_N = algo(N)
f_snag_2N,racoga_2N = algo(2*N)

# plt.subplot(121)
# plt.title("Divergence pour rho trop petits (d=100,N=200)")
# plt.plot(np.log(f_snag_100[0]),label =r"$\rho = 100$",color="b")

# plt.legend()
# plt.xlabel("nb iter")
# plt.ylabel("log(f)")
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(np.log(f_snag_150)[0:2000],label =r"$\lambda = \frac{3}{4}N$",color="violet")
plt.plot(np.log(f_snag_N),color="g",label=r"$\lambda = N$")
# plt.plot(np.log(f_snag_N[1]),color="g",linestyle="--")
# plt.plot(np.log(f_snag_N[2]),color="g",linestyle="--")
print(racoga_150)
print(np.min(racoga_150))
plt.plot(np.log(f_snag_2N),color="r",label=r"$\lambda = 2N$")
label_size = 20
legend_size = 10
number_size = 15
labelpad = 2
plt.xticks((0,1000), fontsize = number_size)
plt.yticks((6,8), fontsize = number_size)
# plt.plot(np.log(f_snag_2N[1]),color="r",linestyle="--")
# plt.plot(np.log(f_snag_2N[2]),color="r",linestyle="--")
plt.xlabel("nb iter",fontsize = label_size, labelpad = labelpad)
plt.ylabel("log(f)",fontsize = label_size, labelpad = labelpad)
plt.legend(fontsize = legend_size)
plt.subplot(122)
plt.hist(racoga_150,label =r"$\lambda = \frac{3}{4}N$",color="violet")
plt.hist(racoga_N,color="g",label=r"$\lambda = N$")
plt.hist(racoga_2N,color="r",label=r"$\lambda = 2N$")
plt.savefig("figures/instability/instability.png")