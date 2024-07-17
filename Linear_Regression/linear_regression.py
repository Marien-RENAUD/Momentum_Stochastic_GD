import numpy as np
import numpy.random as nprandom

rng = np.random.default_rng()

def features_gaussian(d,N,mean,prompt_random_matrix = None,prompt_cov_matrix = None,prompt_bias = None):
    if prompt_bias != None:
        bias = prompt_bias
    else:
        bias = nprandom.normal(nprandom.uniform(-N,N),N,N)
    if prompt_cov_matrix != None:
        cov_matrix = prompt_cov_matrix
        features_matrix = nprandom.multivariate_normal(mean,Cov_matrix,N)
        return features_matrix,b
    else:
        if prompt_random_matrix != None:
            random_matrix = prompt_random_matrix
        else:
            random_matrix = nprandom.uniform(-1,1,(d,d))  
        cov_matrix = np.dot(random_matrix,random_matrix.T)
        features_matrix = nprandom.multivariate_normal(mean,cov_matrix,N)
        return features_matrix,bias
    
def features_gaussian_mixture(d,N,mean,mixture_prob):
    nb_class = len(mixture_prob)
    bias = nprandom.normal(np.zeros(N),N,N)
    random_matrix = nprandom.uniform(-1,1,(nb_class,d,d))    
    cov_matrix = np.matmul(random_matrix,random_matrix.transpose((0,2,1)))
    vec = np.arange(nb_class)
    sample = nprandom.choice(vec,p=mixture_prob,size = N)
    features_matrix = np.empty((N,d))
    k = 0
    for i in range(nb_class):
        nb_ind = len(np.where(sample==i)[0])
        features_matrix[k:(k+nb_ind),:] = nprandom.multivariate_normal(mean[i],cov_matrix[i],nb_ind)
        k += nb_ind
    return features_matrix,bias

def features_gaussian_mixture_det_rep(d,N,mean):
    nb_class = len(mean[:,0])
    if N/batch_size % 1 >0:
        raise Exception("Please chose N and batch_size such that batch_size divide N") 
    bias = nprandom.normal(np.zeros(N),N,(nb_class,N))
    random_matrix = nprandom.uniform(-1,1,(nb_class,d,d))    
    cov_matrix = np.matmul(random_matrix,random_matrix.transpose((0,2,1)))
    features_matrix = np.empty((N,d))
    rep_class = np.empty(nb_class)
    k = 0
    prop = int(N/nb_class)
    for i in range(nb_class):
        features_matrix[k:(k+prop),:] = nprandom.multivariate_normal(mean[i],cov_matrix[i],prop)
        k += prop
    return features_matrix,bias

def features_orthogonal(d,N,prompt_lambda_vec = False, prompt_bias = None):
    if prompt_lambda_vec == False:
        lambda_vec = rng.exponential(1,N)
    else:
        lambda_vec = prompt_lambda_vec
    features_matrix = np.eye(N,d)*lambda_vec.reshape(N,1)
    if prompt_bias != None:
        prompt_bias = nprandom.normal(nprandom.uniform(-N,N),N,N)
    return features_matrix,prompt_bias

def f(x,features_matrix,bias):
    return 0.5*np.mean((np.dot(features_matrix,x)-bias)**2)

def f_sample(x,N,features_matrix,bias):
    return  0.5*( np.mean((np.dot(features_matrix,x)-bias.reshape(N,1))**2,axis = 0))

def grad_f(x,N,features_matrix,bias,return_racoga = False):
    grad =(np.dot(features_matrix,x)-bias).reshape(N,1)*features_matrix
    gradient = np.sum(grad,axis=0)/N
    if return_racoga == True:
        corr = np.dot(grad,grad.T)
        racoga = np.sum(corr[np.triu_indices(N,k=1)])/np.sum(np.diag(corr))
        return gradient,racoga
    return gradient

def grad_sto_f(x,d,N,n_sample,batch_size,features_matrix,bias):
    vec = np.tile(np.arange(N),n_sample).reshape(n_sample,N)
    grad =np.moveaxis((np.dot(features_matrix,x)-bias.reshape(N,1)),0,1).reshape(n_sample,N,1)*features_matrix
    batch_index = rng.permuted(vec,axis=1)[:,:batch_size] 
    gradient = np.take_along_axis(grad,batch_index.reshape(n_sample,batch_size,1),axis=1).mean(axis=1).reshape(n_sample,d).T
    return gradient

def grad_sto_f_batch_rep(x,d,N,n_sample,batch_size,features_matrix,bias,nb_class):
    prop = int(N/nb_class)
    k= 0
    vec = np.arange(k,prop)
    for i in range(1,nb_class):
        k += prop
        vec = np.vstack([vec,np.arange(k,k+ prop)])
    vec_index = rng.permuted(vec,axis=1).transpose((1,0)).flatten()
    for j in range(1,n_sample):
        vec_index = np.vstack([vec_index,rng.permuted(vec,axis=1).transpose((1,0)).flatten()])
    batch_index = vec_index[:,:batch_size] 
    grad =np.moveaxis((np.dot(features_matrix,x)-bias.reshape(N,1)),0,1).reshape(n_sample,N,1)*features_matrix
    gradient = np.take_along_axis(grad,batch_index.reshape(n_sample,batch_size,1),axis=1).mean(axis=1).reshape(n_sample,d).T
    return gradient

def racoga_computation(x,N,features_matrix,bias):
    grad =(np.dot(features_matrix,x)-bias).reshape(N,1)*features_matrix
    corr = np.dot(grad,grad.T)
    racoga = np.sum(corr[np.triu_indices(N,k=1)])/np.sum(np.diag(corr))
    return racoga

def GD(x_0,L,n_iter,d,N,features_matrix,bias,return_racoga = False):
    f_GD = np.empty(n_iter)
    x_GD = np.empty((n_iter,d))
    if return_racoga == True:
        racoga = np.empty(n_iter-1)
    x_GD[0,:] = x_0
    f_GD[0] = f(x_0,features_matrix,bias)

    step = 1/L
    for i in range(n_iter-1):
        if return_racoga == True:
            gradient,racoga[i] = grad_f(x_GD[i],N,features_matrix,bias,return_racoga)
        else:
             gradient = grad_f(x_GD[i],N,features_matrix,bias,return_racoga)
        x_GD[i+1,:] = x_GD[i,:] - step*gradient
        f_GD[i+1] = f(x_GD[i+1,:],features_matrix,bias)
    if return_racoga == True:
        return f_GD,racoga
    else:
        return f_GD

def SGD(x_0,L_sgd,n_iter,n_sample,d,batch_size,N,features_matrix,bias,random_init = False,return_racoga = False):
    x_SGD = np.empty((n_iter,d,n_sample))
    f_SGD = np.empty((n_iter,n_sample))

    if return_racoga == True:
        racoga = np.empty(n_iter-1)

    if random_init == True:
        x_SGD[0,:,:] = nprandom.normal(0, d,(d,n_sample))
        f_sample(x_0,N,features_matrix,bias)
    else:
        x_SGD[0,:,:] = x_0.reshape(d,1)
        f_SGD[0,:] = f(x_0,features_matrix,bias)
   
    step = 1/L_sgd

    for i in range(n_iter-1):
        gradient_sto = grad_sto_f(x_SGD[i,:,:],d,N,n_sample,batch_size,features_matrix,bias)
        if return_racoga == True:
            racoga[i] = racoga_computation(x_SGD[i,:,0],N,features_matrix,bias)
        x_SGD[i+1,:,:] = x_SGD[i,:,:] - step*gradient_sto
        f_SGD[i+1,:] = f_sample(x_SGD[i+1,:],N,features_matrix,bias)

    if return_racoga == True:
        return f_SGD,racoga
    else:
        return f_SGD

def NAG(x_0,mu,L,n_iter,d,N,features_matrix,bias,return_racoga = False):
    x_nag = np.empty((n_iter,d))
    y_nag = np.empty((n_iter,d))
    f_nag = np.empty(n_iter)

    x_nag[0,:] = x_0
    y_nag[0,:] = x_0
    f_nag[0] = f(x_0,features_matrix,bias)

    if return_racoga == True:
        racoga = np.empty(n_iter-1)
    ## Params
    alpha = (1-np.sqrt(mu/L))/(1+np.sqrt(mu/L))
    step = 1/(L)

    for i in range(1,n_iter):
        if return_racoga == True:
            gradient,racoga[i-1] = grad_f(y_nag[i-1],N,features_matrix,bias,return_racoga)
        else:
            gradient = grad_f(y_nag[i-1],N,features_matrix,bias,return_racoga)
        x_nag[i,:] = y_nag[i-1,:] - step*gradient
        y_nag[i,:] = x_nag[i,:] + alpha*(x_nag[i,:] - x_nag[i-1,:])
        f_nag[i] = f(x_nag[i,:],features_matrix,bias)

    if return_racoga == True:
        return f_nag,racoga
    else:
        return f_nag

    
def SNAG(x_0,mu,L,rho,n_iter,n_sample,d,batch_size,N,features_matrix,bias,random_init = False,return_racoga = False):

    x_snag = np.empty((n_iter,d,n_sample))
    y_snag = np.empty((n_iter,d,n_sample))
    z_snag =  np.empty((n_iter,d,n_sample))
    f_snag =  np.empty((n_iter,n_sample))

    if return_racoga == True:
        racoga = np.empty(n_iter-1)

    if random_init == False:
        x_snag[0,:,:] = x_0.reshape(d,1)
        y_snag[0,:,:] = x_0.reshape(d,1)
        z_snag[0,:,:] = x_0.reshape(d,1)
        f_snag[0,:] =  f(x_0,features_matrix,bias)

    else:
        x_0 = nprandom.normal(0, d,(d,n_sample))
        x_snag[0,:,:] =  x_0
        y_snag[0,:,:] = x_0
        z_snag[0,:,:] =  x_0
        f_snag[0,:] =  f_sample(x_0,N,features_matrix,bias)

    # Params
    step = 1/(L*rho)
    eta = 1/(np.sqrt(mu*L)*rho)
    beta = 1-np.sqrt( (mu/L) )/rho
    alpha = 1/(1+(1/rho)*np.sqrt(mu/L))
    for i in range(1,n_iter):
        gradient_sto = grad_sto_f(y_snag[i-1,:,:],d,N,n_sample,batch_size,features_matrix,bias)
        if return_racoga == True:
            racoga[i-1] = racoga_computation(y_snag[i-1,:,0],N,features_matrix,bias)
        x_snag[i,:,:] = y_snag[i-1,:,:] - step*gradient_sto
        z_snag[i,:,:] = beta*z_snag[i-1,:,:] + (1-beta)*y_snag[i-1,:,:] - eta*gradient_sto
        y_snag[i,:,:] = z_snag[i,:,:]*(1-alpha) + (alpha)*x_snag[i,:,:]

        f_snag[i,:] =f_sample(x_snag[i,:],N,features_matrix,bias)

    if return_racoga == True:
        return f_snag,racoga
    else:
        return f_snag