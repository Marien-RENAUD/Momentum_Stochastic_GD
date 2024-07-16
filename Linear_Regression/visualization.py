import numpy as np
import matplotlib.pyplot as plt
import os as os
print(os.getcwd() )

case = 2 # 0 : overparameterized, 1 : d=N, 2 : underparameterized
path_figure_root_cv= 'convergence_linear_regression_'
path_figure_root_racoga  = 'racoga_'
if case == 0:
    suffixe = 'overparameterized'
    path_figure_cv = 'results/overparameterized/' + path_figure_root_cv +' convergence_linear_regression_'+ 'overparameterized.png'
    path_figure_racoga = 'results/overparameterized/' + path_figure_root_racoga + 'overparameterized.png'
    param= np.load("param_" + suffixe +".npy",allow_pickle=True).item()
    algo = np.load("algo_" + suffixe +".npy",allow_pickle=True).item()
    racoga = np.load("racoga_" + suffixe +".npy",allow_pickle=True).item()
elif case == 1:
    suffixe = 'd=N'
    path_figure_cv = 'results/d=N/' + path_figure_root_cv + 'convergence_linear_regression_'+ 'd_N.png'
    path_figure_racoga = 'results/d=N/' + path_figure_root_racoga + 'd=N.png'
    param= np.load("param_" + suffixe +".npy",allow_pickle=True).item()
    algo = np.load("algo_" + suffixe +".npy",allow_pickle=True).item()
    racoga = np.load("racoga_" + suffixe +".npy",allow_pickle=True).item()
else:
    suffixe = 'underparameterized'
    path_figure_cv = 'results/underparameterized/' + path_figure_root_cv + 'convergence_linear_regression_' + 'underparameterized.png'
    path_figure_racoga = 'results/underparameterized/' + path_figure_root_racoga + 'underparameterized.png' 
    param= np.load("param_" + suffixe +".npy",allow_pickle=True).item()
    algo = np.load("algo_" + suffixe +".npy",allow_pickle=True).item()
    racoga = np.load("racoga_" + suffixe +".npy",allow_pickle=True).item()

d, N , n_iter, batch_size ,mu , L, L_max = param["d"], param["N"],param["n_iter"], param["batch_size"], param["mu"], param["L"], param["L_max"]

labels = np.load("labels.npy")
index = np.load("index.npy")
plt.figure(figsize=(20,10))
nb_alg = len(labels) 
nb_rho = len(param["rho"])
nb_gd_eval = np.arange(0,n_iter+1,int(N/batch_size))



k = 0
for j in range(nb_alg):
    if j==0:
        col = "black"
    elif j ==1:
        col = "grey"
    elif j == 2:
        col = "red"
    else:
        col = (0.25, k/nb_rho ,1-k/nb_rho)
        k+=1
    if len(algo[index[j]].shape) >1:
        mean_alg,min_alg,max_alg = np.mean(algo[index[j]],axis=1),np.min(algo[index[j]],axis=1),np.max(algo[index[j]],axis=1)
        plt.plot(np.log(mean_alg),label=labels[j],color =col,lw=2)
        plt.plot(np.log(min_alg),color =col,linestyle ="--")
        plt.plot(np.log(max_alg),color =col,linestyle ="--")
    else:
        plt.plot(nb_gd_eval,np.log(algo[index[j]]),label=labels[j],color =col,lw=2)
plt.xlabel("Nb gradient evaluations",fontsize = 13)
plt.ylabel(r"$\log(f)$",fontsize= 13)
plt.legend()
plt.title("Algorithms convergence, N = " + str(N) + ", d = " + str(d))
plt.savefig(path_figure_cv)
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.yticks((racoga["gd"].min(),0,racoga["gd"].max()))
plt.plot(racoga["gd"],label = "GD",color="b")
plt.plot(racoga["nag"],label = "NAG",color="r")
plt.title("RACOGA condition along iterations",fontsize = 10)
plt.legend()
plt.subplot(122)
for j in range(nb_rho):
    col = (0.5, j/nb_alg ,1-j/nb_alg)
    plt.hist(racoga[index[j+3]],bins=np.linspace(racoga[index[j+3]].min(),racoga[index[j+3]].max(),50),edgecolor="black",facecolor = col,label = labels[j+3])
plt.title("RACOGA condition number along iterations of SNAG",fontsize=10)
plt.legend()
plt.savefig(path_figure_racoga)