import numpy as np 
import matplotlib.pyplot as plt
import numpy.random as nprandom
import numpy.linalg as nplinalg

### Dimension, Nb of functions
d,N = 10,10

### Initialisation features
Random_matrix = nprandom.uniform(-1,1,(d,d))
Cov_matrix = np.dot(Random_matrix,Random_matrix.T)
# Cov_matrix = np.eye(d)
VEC_RAND = nprandom.multivariate_normal(np.zeros(d),Cov_matrix,N)
# VEC_RAND = nprandom.multivariate_normal(nprandom.uniform(-d,d,d),Cov_matrix,N)
b = nprandom.normal(nprandom.uniform(N),N,N)

x_0 = nprandom.normal(0,10,d)

### Nb iterations, Nb sample
n_iter_gen= 1*10**5
# n_iter_gen = 2*N
n_sample = 10


# x_0 = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[1][:,min_vp]
GRAD =(np.dot(VEC_RAND,x_0)-b).reshape(N,1)*VEC_RAND
### GD ### 
n_iter=int(n_iter_gen/N)+1
f_GD = np.empty(n_iter)
x_GD = np.empty((n_iter,d))
x_GD[0,:] = x_0

f_GD[0] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_0)-b)**2))
step = N/nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].max()
for i in range(n_iter-1):
    GRAD =(np.dot(VEC_RAND,x_GD[i,:])-b).reshape(N,1)*VEC_RAND
    gradi = np.sum(GRAD,axis=0)/N
    x_GD[i+1,:] = x_GD[i,:] - step*gradi
    f_GD[i+1] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_GD[i+1,:])-b)**2))

### SGD ###

GRAD =(np.dot(VEC_RAND,x_0)-b).reshape(N,1)*VEC_RAND
x_SGD = np.empty((n_iter_gen,d,n_sample))
f_SGD = np.empty((n_iter_gen,n_sample))
x_SGD[0,:,:] = x_0.reshape(d,1)
f_SGD[0,:] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_SGD[0,:])-b.reshape(N,1))**2,axis = 0))
step = 1/nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].max()

for i in range(n_iter_gen-1):
    GRAD =np.moveaxis((np.dot(VEC_RAND,x_SGD[i,:,:])-b.reshape(N,1)),0,1).reshape(n_sample,N,1)*VEC_RAND
    U = nprandom.randint(0,N,n_sample)
    x_SGD[i+1,:,:] = x_SGD[i,:,:] - step*np.take_along_axis(GRAD,U.reshape(n_sample,1,1),axis=1).reshape(n_sample,d).T
    f_SGD[i+1,:] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_SGD[i+1,:])-b.reshape(N,1))**2,axis = 0))

### NESTEROV DETERMINISTE ###
n_iter=int(n_iter_gen/N)+1
x_nag = np.empty((n_iter,d))
y_nag = np.empty((n_iter,d))
f_nag = np.empty(n_iter)

x_nag[0,:] = x_0
y_nag[0,:] = x_0
f_nag[0] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_0)-b)**2))

## Params
mu = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].min()/N
L = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].max()/N
alpha = (1-np.sqrt(mu/L))/(1+np.sqrt(mu/L))
step = 1/(L)
## First step

count=0
for i in range(1,n_iter):
    GRAD =(np.dot(VEC_RAND,y_nag[i-1,:])-b).reshape(N,1)*VEC_RAND
    gradient = np.sum(GRAD,axis=0)/N
    x_nag[i,:] = y_nag[i-1,:] - step*gradient
    y_nag[i,:] = x_nag[i,:] + alpha*(x_nag[i,:] - x_nag[i-1,:])
    f_nag[i] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_nag[i,:])-b)**2))
### STOCHASTIC NESTEROV ###

## rho = N/2

x_snag = np.empty((n_iter_gen,d,n_sample))
y_snag = np.empty((n_iter_gen,d,n_sample))
z_snag =  np.empty((n_iter_gen,d,n_sample))
f_snag_bis =  np.empty((n_iter_gen,n_sample))

x_snag[0,:,:] = x_0.reshape(d,1)
y_snag[0,:,:] = x_0.reshape(d,1)
z_snag[0,:,:] = x_0.reshape(d,1)
f_snag_bis[0,:] =  1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_snag[0,:])-b.reshape(N,1))**2,axis = 0))

# ## Params

mu = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].min()/N
L = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].max()/N
rho=N/2

step = 1/(L*rho)
eta = 1/(np.sqrt(mu*L)*rho)
beta = 1-np.sqrt( (mu/L) )/rho
alpha = 1/(1+(1/rho)*np.sqrt(mu/L))
for i in range(1,n_iter_gen):
    GRAD =np.moveaxis((np.dot(VEC_RAND,y_snag[i-1,:,:])-b.reshape(N,1)),0,1).reshape(n_sample,N,1)*VEC_RAND
    U = nprandom.randint(0,N,n_sample)

    x_snag[i,:,:] = y_snag[i-1,:,:] - step*np.take_along_axis(GRAD,U.reshape(n_sample,1,1),axis=1).reshape(n_sample,d).T
    z_snag[i,:,:] = beta*z_snag[i-1,:,:] + (1-beta)*y_snag[i-1,:,:] - eta*np.take_along_axis(GRAD,U.reshape(n_sample,1,1),axis=1).reshape(n_sample,d).T
    y_snag[i,:,:] = z_snag[i,:,:]*(1-alpha) + (alpha)*x_snag[i,:,:]

    f_snag_bis[i,:] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_snag[i,:])-b.reshape(N,1))**2,axis = 0))

## rho = N
f_snag =  np.empty((n_iter_gen,n_sample))
f_snag[0,:] =  1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_snag[0,:])-b.reshape(N,1))**2,axis = 0))

# ## Params

mu = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].min()/N
L = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].max()/N
rho=N
# eta = 1/(rho*L)
# step = 1/(L*rho)
# gamma = 1/np.sqrt(mu*eta*rho)
# beta = 1-np.sqrt( (mu*eta)/rho )
step = 1/(L*rho)
eta = 1/(np.sqrt(mu*L)*rho)
beta = 1-np.sqrt( (mu/L))/rho
alpha = 1/(1+(1/rho)*np.sqrt(mu/L))
for i in range(1,n_iter_gen):
    GRAD =np.moveaxis((np.dot(VEC_RAND,y_snag[i-1,:,:])-b.reshape(N,1)),0,1).reshape(n_sample,N,1)*VEC_RAND
    U = nprandom.randint(0,N,n_sample)

    x_snag[i,:,:] = y_snag[i-1,:,:] - step*np.take_along_axis(GRAD,U.reshape(n_sample,1,1),axis=1).reshape(n_sample,d).T
    z_snag[i,:,:] = beta*z_snag[i-1,:,:] + (1-beta)*y_snag[i-1,:,:] - eta*np.take_along_axis(GRAD,U.reshape(n_sample,1,1),axis=1).reshape(n_sample,d).T
    y_snag[i,:,:] = z_snag[i,:,:]*(1-alpha) + (alpha)*x_snag[i,:,:]

    f_snag[i,:] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_snag[i,:])-b.reshape(N,1))**2,axis = 0))

## rho = 2N
f_snag_bisbis =  np.empty((n_iter_gen,n_sample))
f_snag_bisbis[0,:] =  1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_snag[0,:])-b.reshape(N,1))**2,axis = 0))

# ## Params

mu = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].min()/N
L = nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0].max()/N
rho=2*N
# eta = 1/(rho*L)
# step = 1/(L*rho)
# gamma = 1/np.sqrt(mu*eta*rho)

step = 1/(L*rho)
eta = 1/(np.sqrt(mu*L)*rho)
beta = 1-np.sqrt((mu/L))/rho
alpha = 1/(1+(1/rho)*np.sqrt(mu/L))
for i in range(1,n_iter_gen):
    GRAD =np.moveaxis((np.dot(VEC_RAND,y_snag[i-1,:,:])-b.reshape(N,1)),0,1).reshape(n_sample,N,1)*VEC_RAND
    U = nprandom.randint(0,N,n_sample)

    x_snag[i,:,:] = y_snag[i-1,:,:] - step*np.take_along_axis(GRAD,U.reshape(n_sample,1,1),axis=1).reshape(n_sample,d).T
    z_snag[i,:,:] = beta*z_snag[i-1,:,:] + (1-beta)*y_snag[i-1,:,:] - eta*np.take_along_axis(GRAD,U.reshape(n_sample,1,1),axis=1).reshape(n_sample,d).T
    y_snag[i,:,:] = z_snag[i,:,:]*(1-alpha) + alpha*x_snag[i,:,:]

    f_snag_bisbis[i,:] = 1/N * (0.5 * np.sum((np.dot(VEC_RAND,x_snag[i,:])-b.reshape(N,1))**2,axis = 0))

print("Conditionning : ",(nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0]).min()/(nplinalg.eig(np.dot(VEC_RAND,VEC_RAND.T))[0]).max())
mat_angles = (np.dot(VEC_RAND,VEC_RAND.T)/np.sqrt(np.diag(np.dot(VEC_RAND,VEC_RAND.T))))/np.sqrt(np.diag(np.dot(VEC_RAND,VEC_RAND.T))).reshape(N,1)
array_angles = np.abs(mat_angles[np.triu_indices(N,k=1)])
print("moy Angles coeffs : ", np.mean(array_angles), "STD angles coeffs : ", np.std(array_angles), "min, max angles coeffs : ", np.min(array_angles), np.max(array_angles))
vec_norm= (VEC_RAND**2).sum(axis=1)
print("Variance des normes des features : " ,np.std(vec_norm))
print("Conditionnement des normes des features : ", vec_norm.max()/vec_norm.min())
print(vec_norm, "L rho  = ", L*0.5*N)
print( vec_norm.max() < L*0.5*N)
plt.figure(figsize=(20,10))
# plt.subplot(121)
# plt.hist(array_angles)
# plt.subplot(122)
plt.title("Convergence speed")
nb_gd_eval = np.arange(0,n_iter_gen+1,N)
plt.plot(nb_gd_eval,np.log(f_GD),label="GD",color="black")

mean_sgd = np.mean(f_SGD,axis=1)
sd_sgd = np.std(f_SGD,axis=1)
min_sgd = np.min(f_SGD,axis=1)
max_sgd = np.max(f_SGD,axis=1)
plt.plot(np.log(mean_sgd),label="mean-SGD",color ="grey",lw=2)
# # plt.plot(np.log(mean_sgd+ sd_sgd),label="ec-SGD",color ="orange",linestyle ="--")
# # plt.plot(np.log(mean_sgd- sd_sgd),color ="orange",linestyle ="--")
plt.plot(np.log(min_sgd),color ="grey",linestyle ="--")
plt.plot(np.log(max_sgd),color ="grey",linestyle ="--")

plt.plot(nb_gd_eval,np.log(f_nag),label="NAG",color="r")
linestyle = (0,(5,10))

mean_snag_bis = np.mean(f_snag_bis,axis=1)
min_snag_bis = np.min(f_snag_bis,axis=1)
max_snag_bis = np.max(f_snag_bis,axis=1)
plt.plot(np.log(mean_snag_bis),color="orange",label="SNAG rho=N/2")
plt.plot(np.log(min_snag_bis),color="orange",linestyle =linestyle,alpha = 0.5)
plt.plot(np.log(max_snag_bis),color="orange",linestyle =linestyle,alpha = 0.5)


mean_snag = np.mean(f_snag,axis=1)
sd_snag = np.std(f_snag,axis=1)
min_snag = np.min(f_snag,axis=1)
max_snag = np.max(f_snag,axis=1)
plt.plot(np.log(mean_snag),color="g",label="SNAG rho=N")
plt.plot(np.log(min_snag),color="g",linestyle =linestyle,alpha = 0.5)
plt.plot(np.log(max_snag),color="g",linestyle =linestyle,alpha = 0.5)


mean_snag_bisbis = np.mean(f_snag_bisbis,axis=1)
min_snag_bisbis = np.min(f_snag_bisbis,axis=1)
max_snag_bisbis = np.max(f_snag_bisbis,axis=1)
plt.plot(np.log(mean_snag_bisbis),color="b",label="SNAG rho=2N")
plt.plot(np.log(min_snag_bisbis),color="b",linestyle =linestyle,alpha = 0.5)
plt.plot(np.log(max_snag_bisbis),color="b",linestyle =linestyle,alpha = 0.5)


plt.ylabel(r"$log(f)$")
plt.xlabel("Nb gradients evaluations")
plt.legend()


