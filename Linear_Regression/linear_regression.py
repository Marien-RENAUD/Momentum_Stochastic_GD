import torch
import numpy as np
import numpy.random as nprandom

rng = np.random.default_rng()
# torch.set_default_dtype(torch.float64)
def features_gaussian(d, N, mean, generate_bias=False):
    if generate_bias:
        bias = torch.normal(torch.rand(N) * 2 * N - N, torch.tensor([N]))
    else:
        bias = None
    
    random_matrix = torch.rand((d, d)) * 2 - 1
    cov_matrix = torch.matmul(random_matrix, random_matrix.t())
    features_matrix = torch.distributions.MultivariateNormal(mean, scale_tril= torch.tril(cov_matrix))
    return features_matrix.sample((N,)), bias

def features_orthogonal(d,N,generate_lambda = False, generate_bias = False):
    if generate_lambda:
        lambda_vec = torch.distributions.Exponential(1).sample((N,))
        features_matrix = torch.eye(N,d)*lambda_vec.reshape(N,1)
    else:
        features_matrix = torch.eye(N,d)
    if generate_bias :
        bias = torch.normal(torch.rand(N) * 2 * N - N, torch.tensor([N]))
        return features_matrix,bias
    return features_matrix, None

def features_gaussian_mixture(d,N,mean,mixture_prob): # features of law gaussian mixture
    nb_class = len(mixture_prob)
    bias = torch.normal(torch.rand(N) * 2 * N - N, torch.tensor([N]))
    random_matrix = torch.rand((nb_class,d, d)) * 2 - 1   
    cov_matrix = torch.matmul(random_matrix,random_matrix.transpose(1,2))
    vec = np.arange(nb_class)
    sample = nprandom.choice(vec,p=mixture_prob,size = N)
    features_matrix = torch.empty((N,d))
    k = 0
    for i in range(nb_class):
        nb_ind = len(np.where(sample==i)[0])
        features_matrix[k:(k+nb_ind),:] = torch.distributions.MultivariateNormal(mean[i],scale_tril = torch.tril(cov_matrix[i])).sample((nb_ind,))
        k += nb_ind
    return features_matrix,bias

def features_gaussian_mixture_det_rep(d,N,mean): # features of law gaussian mixture with equal and sorted repartition of each "class"
    nb_class = len(mean[:,0])
    if N/nb_class % 1 >0:
        raise Exception("Please chose N and number of cluster such that number of cluster divide N") 
    bias = torch.normal(torch.rand(N) * 2 * N - N, torch.tensor([N]))
    random_matrix = torch.rand((nb_class,d, d)) * 2 - 1   
    cov_matrix = torch.matmul(random_matrix,random_matrix.transpose(1,2))
    features_matrix = torch.empty((N,d))
    rep_class = np.empty(nb_class)
    k = 0
    prop = int(N/nb_class)
    for i in range(nb_class):
        features_matrix[k:(k+prop),:] = torch.distributions.MultivariateNormal(mean[i],scale_tril = torch.tril(cov_matrix[i])).sample((prop,))
        k += prop
    return features_matrix,bias

def f(x, features_matrix, bias):
    return 0.5 * torch.mean((torch.matmul(features_matrix, x) - bias) ** 2)

def f_i(x, index, features_matrix, bias):
    return 0.5 * (torch.matmul(features_matrix[index, :], x) - bias[index]) ** 2

def f_sample(x, N, features_matrix, bias):
    return 0.5 * torch.mean((torch.matmul(features_matrix, x) - bias.view(N, 1)) ** 2, dim=0)

def grad_f(x,N,features_matrix,bias,return_racoga = False):
    grad =(torch.matmul(features_matrix,x)-bias).reshape(N,1)*features_matrix
    gradient = torch.sum(grad,axis=0)/N
    if return_racoga == True:
        corr = torch.matmul(grad,grad.T)

        racoga = torch.sum(torch.triu(corr,diagonal=1))/torch.sum(torch.diag(corr))
        return gradient,racoga
    return gradient

def grad_sto_f(x, d, N, n_sample, batch_size, features_matrix, bias):
    # Création du vecteur 'vec'
    vec = np.tile(np.arange(N),n_sample).reshape(n_sample,N)

    # Calcul de 'grad'
    grad = torch.moveaxis((torch.matmul(features_matrix, x) - bias.reshape(N, 1)), 0, 1).view(n_sample, N, 1) * features_matrix
    # Sélection aléatoire des index de batch

    batch_index = torch.tensor(rng.permuted(vec,axis=1)[:,:batch_size])
    # Calcul du gradient final
    gradient = torch.take_along_dim(grad, batch_index.view(n_sample, batch_size, 1), dim=1).mean(dim=1).T
    
    return gradient 

def grad_sto_f_batch_rep(x,d,N,n_sample,batch_size,features_matrix,bias,nb_class):
    prop = int(N/nb_class)
    k= 0
    vec = torch.arange(k,prop)
    for i in range(1,nb_class):
        k += prop
        vec = np.vstack([vec,np.arange(k,k+ prop)])
    vec_index = rng.permuted(vec,axis=1).transpose((1,0)).flatten()
    for j in range(1,n_sample):
        vec_index = np.vstack([vec_index,rng.permuted(vec,axis=1).transpose((1,0)).flatten()])
    batch_index = torch.tensor(vec_index[:,:batch_size])
    grad = torch.moveaxis((torch.matmul(features_matrix, x) - bias.reshape(N, 1)), 0, 1).view(n_sample, N, 1) * features_matrix
    gradient = torch.take_along_dim(grad, batch_index.view(n_sample, batch_size, 1), dim=1).mean(dim=1).T
    return gradient

def racoga_computation(x,N,features_matrix,bias):
    grad =(torch.matmul(features_matrix,x)-bias).reshape(N,1)*features_matrix
    corr = torch.matmul(grad,grad.T)
    racoga = torch.sum(torch.triu(corr,diagonal=1))/torch.sum(torch.diag(corr))
    # return N/(1+2*racoga)
    return racoga
    
    
def racoga_computation_alternative_sampling(x,N,features_matrix,bias):
    grad =(torch.matmul(features_matrix,x)-bias).reshape(N,1)*features_matrix
    corr = torch.matmul(grad,grad.T)
    racoga = torch.sum(torch.triu(corr,diagonal=1))/torch.sum(torch.diag(corr))
    return N/(1+2*racoga)    


def GD(x_0,L,n_iter,d,N,features_matrix,bias,return_racoga = False):
    f_GD = torch.empty(n_iter)
    x_GD = torch.empty((n_iter,d))
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
            
def SGD(x_0, L_sgd, n_iter, n_sample, d, batch_size, N, features_matrix, bias, random_init=False, return_racoga=False, alternative_sampling=False, nb_class=None):
    x_SGD = torch.empty((n_iter, d, n_sample))
    f_SGD = torch.empty((n_iter, n_sample))

    if return_racoga:
        racoga = torch.empty(n_iter - 1)


    if random_init:
        x_SGD[0, :, :] = torch.normal(0, 1, size=(d, n_sample))  # Using standard normal distribution for initialization
        f_sample(x_0_torch, N, features_matrix, bias)
    else:
        x_SGD[0, :, :] = x_0.reshape(d,1)
        f_SGD[0, :] = f(x_0, features_matrix, bias)

    step = 1 / L_sgd

    for i in range(n_iter - 1):
        if not alternative_sampling:
            gradient_sto = grad_sto_f(x_SGD[i, :, :], d, N, n_sample, batch_size, features_matrix, bias)
        else:
            gradient_sto = grad_sto_f_batch_rep(x_SGD[i, :, :], d, N, n_sample, batch_size, features_matrix, bias, nb_class)
        if return_racoga:
            racoga[i] = racoga_computation(x_SGD[i, :, 0], N, features_matrix, bias)
        # grad =(torch.matmul(features_matrix,x_SGD[i, :, :])-bias)[:,0].reshape(N,1)*features_matrix
        # gradient_sto = grad[int(i%N),:].reshape(N,1)
        x_SGD[i + 1, :, :] = x_SGD[i, :, :] - step * gradient_sto
        f_SGD[i + 1, :] = f_sample(x_SGD[i + 1, :, :], N, features_matrix, bias)

    if return_racoga:
        return f_SGD, racoga
    else:
        return f_SGD  
def NAG(x_0,mu,L,n_iter,d,N,features_matrix,bias,return_racoga = False):
    x_nag = torch.empty((n_iter,d))
    y_nag = torch.empty((n_iter,d))
    f_nag = torch.empty(n_iter)

    x_nag[0,:] = x_0
    y_nag[0,:] = x_0
    f_nag[0] = f(x_0,features_matrix,bias)

    if return_racoga == True:
        racoga = torch.empty(n_iter-1)
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

def SNAG(x_0,mu,L,rho,n_iter,n_sample,d,batch_size,N,features_matrix,bias,random_init = False,return_racoga = False,alternative_sampling = False,nb_class = None,L_max = None):

    x_snag = torch.empty((n_iter,d,n_sample))
    y_snag = torch.empty((n_iter,d,n_sample))
    z_snag =  torch.empty((n_iter,d,n_sample))
    f_snag =  torch.empty((n_iter,n_sample))

    if return_racoga == True:
        racoga = torch.empty(n_iter-1)

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
    step = 1/L_max
    eta = 1/(np.sqrt(mu*L)*rho)
    beta = 1-np.sqrt( (mu/L) )/rho
    alpha = 1/(1+(1/rho)*np.sqrt(mu/L))
    for i in range(1,n_iter):
        if alternative_sampling == False:
            gradient_sto = grad_sto_f(y_snag[i-1,:,:],d,N,n_sample,batch_size,features_matrix,bias)
        else:
            gradient_sto = grad_sto_f_batch_rep(y_snag[i-1,:,:],d,N,n_sample,batch_size,features_matrix,bias,nb_class)
        if return_racoga == True:
            racoga[i-1] = racoga_computation(y_snag[i-1,:,0],N,features_matrix,bias)
        # grad =(torch.matmul(features_matrix,y_snag[i-1,:,:])-bias)[:,0].reshape(N,1)*features_matrix
        # gradient_sto = grad[int(i%N),:].reshape(N,1)
        x_snag[i,:,:] = y_snag[i-1,:,:] - step*gradient_sto
        z_snag[i,:,:] = beta*z_snag[i-1,:,:] + (1-beta)*y_snag[i-1,:,:] - eta*gradient_sto
        y_snag[i,:,:] = z_snag[i,:,:]*(1-alpha) + (alpha)*x_snag[i,:,:]

        f_snag[i,:] =f_sample(x_snag[i,:],N,features_matrix,bias)

    if return_racoga == True:
        return f_snag,racoga
    else:
        return f_snag          