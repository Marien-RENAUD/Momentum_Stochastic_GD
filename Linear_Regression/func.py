
def features(d,N,prompt_random_matrix = None,prompt_cov_matrix = None,prompt_bias = None):
    if prompt_bias != None:
        b = prompt_bias
    else:
        b = nprandom.normal(nprandom.uniform(-N,N),N,N)
    if prompt_cov_matrix != None:
        cov_matrix = prompt_cov_matrix
        features_matrix = nprandom.multivariate_normal(np.zeros(d),Cov_matrix,N)
        return features_matrix,b
    else:
        if prompt_random_matrix != None:
            random_matrix = prompt_random_matrix
        else:
            random_matrix = nprandom.uniform(-1,1,(d,d))  
        cov_matrix = np.dot(random_matrix,random_matrix.T)
        features_matrix = nprandom.multivariate_normal(np.zeros(d),cov_matrix,N)
        return features_matrix,b
def f(x,features_matrix,bias):
    return 0.5*np.mean((np.dot(features_matrix,x)-bias)**2)
def f_sample(x,N,features_matrix,bias):
    return  0.5*( np.mean((np.dot(features_matrix,x)-bias.reshape(N,1))**2,axis = 0))
def grad_f(x,N,features_matrix,bias):
    grad =(np.dot(features_matrix,x)-bias).reshape(N,1)*features_matrix
    gradient = np.sum(grad,axis=0)/N
    return gradient
def grad_sto_f(x,N,batch_size,features_matrix,bias):
    vec = np.tile(np.arange(N),n_sample).reshape(n_sample,N)
    grad =np.moveaxis((np.dot(features_matrix,x)-bias.reshape(N,1)),0,1).reshape(n_sample,N,1)*features_matrix
    batch_index = rng.permuted(vec,axis=1)[:,:batch_size] 
    gradient = np.take_along_axis(grad,batch_index.reshape(n_sample,batch_size,1),axis=1).mean(axis=1).reshape(n_sample,d).T
    return gradient
def GD(x_0,L,n_iter,d,N,features_matrix,bias):
    f_GD = np.empty(n_iter)
    x_GD = np.empty((n_iter,d))

    x_GD[0,:] = x_0
    f_GD[0] = f(x_0,features_matrix,bias)

    step = 1/L
    for i in range(n_iter-1):
        gradient = grad_f(x_GD[i],N,features_matrix,bias)
        x_GD[i+1,:] = x_GD[i,:] - step*gradient
        f_GD[i+1] = f(x_GD[i+1,:],features_matrix,bias)
    return f_GD

def SGD(x_0,L_max,n_iter,n_sample,d,batch_size,N,features_matrix,bias):
    x_SGD = np.empty((n_iter,d,n_sample))
    f_SGD = np.empty((n_iter,n_sample))
    x_SGD[0,:,:] = x_0.reshape(d,1)
    f_SGD[0,:] = f(x_0,features_matrix,bias)
    step = 1/L_max

    for i in range(n_iter-1):
        gradient_sto = grad_sto_f(x_SGD[i,:,:],N,batch_size,features_matrix,bias)
        x_SGD[i+1,:,:] = x_SGD[i,:,:] - step*gradient_sto
        f_SGD[i+1,:] = f_sample(x_SGD[i+1,:],N,features_matrix,bias)
    return f_SGD
def NAG(x_0,mu,L,n_iter,d,N,features_matrix,bias):
    x_nag = np.empty((n_iter,d))
    y_nag = np.empty((n_iter,d))
    f_nag = np.empty(n_iter)

    x_nag[0,:] = x_0
    y_nag[0,:] = x_0
    f_nag[0] = f(x_0,features_matrix,bias)

    ## Params
    alpha = (1-np.sqrt(mu/L))/(1+np.sqrt(mu/L))
    step = 1/(L)

    for i in range(1,n_iter):
        gradient = grad_f(y_nag[i-1],N,features_matrix,bias)
        x_nag[i,:] = y_nag[i-1,:] - step*gradient
        y_nag[i,:] = x_nag[i,:] + alpha*(x_nag[i,:] - x_nag[i-1,:])
        f_nag[i] = f(x_nag[i,:],features_matrix,bias)

    return f_nag
    
def SNAG(x_0,mu,L,rho,n_iter,n_sample,d,batch_size,N,features_matrix,bias):
    x_snag = np.empty((n_iter_gen,d,n_sample))
    y_snag = np.empty((n_iter_gen,d,n_sample))
    z_snag =  np.empty((n_iter_gen,d,n_sample))
    f_snag =  np.empty((n_iter_gen,n_sample))

    x_snag[0,:,:] = x_0.reshape(d,1)
    y_snag[0,:,:] = x_0.reshape(d,1)
    z_snag[0,:,:] = x_0.reshape(d,1)
    f_snag[0,:] =  f(x_0,features_matrix,bias)

    # Params
    step = 1/(L*rho)
    eta = 1/(np.sqrt(mu*L)*rho)
    beta = 1-np.sqrt( (mu/L) )/rho
    alpha = 1/(1+(1/rho)*np.sqrt(mu/L))
    for i in range(1,n_iter_gen):
        gradient_sto = grad_sto_f(y_snag[i-1,:,:],N,batch_size,features_matrix,bias)

        x_snag[i,:,:] = y_snag[i-1,:,:] - step*gradient_sto
        z_snag[i,:,:] = beta*z_snag[i-1,:,:] + (1-beta)*y_snag[i-1,:,:] - eta*gradient_sto
        y_snag[i,:,:] = z_snag[i,:,:]*(1-alpha) + (alpha)*x_snag[i,:,:]

        f_snag[i,:] =f_sample(x_snag[i,:],N,features_matrix,bias)
    return f_snag