import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
print(os.getcwd() )

version_bis = False # Set true to not overwrite the first experiment
root = "simul_data/" # Data results folder

case = np.load("case.npy") # 0 : overparameterized, 1 : d=N, 2 : underparameterized
biased_features = np.load("biased_features.npy") # True : biased features

if biased_features == True:
    path_figure_root= 'results/biased_features'
else:
    path_figure_root= 'results/unbiased_features'


if case == 0:
    suffixe = 'overparameterized'
    if biased_features == True:
        suffixe += '_biased_features'
    else:
        suffixe += '_unbiased_features'
    if version_bis == True:
        suffixe += '_bis'
    path_figure_cv = path_figure_root + '/overparameterized/convergence_linear_regression_' + suffixe + '.png'
    path_figure_racoga = path_figure_root +'/overparameterized/racoga_' + suffixe + '.png' 
    param= np.load(root +"param_" + suffixe +".npy",allow_pickle=True).item()
    algo = np.load(root +"algo_" + suffixe +".npy",allow_pickle=True).item()
    racoga = np.load(root +"racoga_" + suffixe +".npy",allow_pickle=True).item()
elif case == 1:
    suffixe = 'd=N'
    if biased_features == True:
        suffixe += '_biased_features'
    else:
        suffixe += '_unbiased_features'
    if version_bis == True:
        suffixe += '_bis'
    path_figure_cv = path_figure_root + '/d=N/convergence_linear_regression_' + suffixe + '.png'
    path_figure_racoga = path_figure_root + '/d=N/racoga_' + suffixe + '.png' 
    param= np.load(root +"param_" + suffixe +".npy",allow_pickle=True).item()
    algo = np.load(root +"algo_" + suffixe +".npy",allow_pickle=True).item()
    racoga = np.load(root +"racoga_" + suffixe +".npy",allow_pickle=True).item()
else:
    suffixe = 'underparameterized'
    if biased_features == True:
        suffixe += '_biased_features'
    else:
        suffixe += '_unbiased_features'
    if version_bis == True:
        suffixe += '_bis'
    path_figure_cv = path_figure_root +'/underparameterized/convergence_linear_regression_' + suffixe + '.png'
    path_figure_racoga = path_figure_root +'/underparameterized/racoga_' + suffixe + '.png' 
    param= np.load(root +"param_" + suffixe +".npy",allow_pickle=True).item()
    algo = np.load(root +"algo_" + suffixe +".npy",allow_pickle=True).item()
    racoga = np.load(root +"racoga_" + suffixe +".npy",allow_pickle=True).item()

d, N , n_iter, batch_size ,mu , L, L_max = param["d"], param["N"],param["n_iter"], param["batch_size"], param["mu"], param["L"], param["L_max"]

labels = np.load(root + "labels.npy")
index = np.load(root + "index.npy")
plt.figure(figsize=(20,10))
nb_alg = len(labels) 
nb_rho = len(param["rho"])
nb_gd_eval = np.arange(0,n_iter+1,int(N/batch_size))


racoga_mean, racoga_median, racoga_decile_inf, racoga_quantile_01, racoga_min, racoga_max = np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg)
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
    racoga_current = racoga[index[j]]
    racoga_mean[j],racoga_median[j], racoga_decile_inf[j], racoga_quantile_01[j], racoga_min[j], racoga_max[j] = racoga_current.mean(), np.quantile(racoga_current,0.5,method = 'nearest'), np.quantile(racoga_current,0.1,method = 'nearest'), np.quantile(racoga_current,0.01,method = 'nearest'), racoga_current.min(), racoga_current.max()
vec_racoga = np.vstack((racoga_mean, racoga_median, racoga_decile_inf, racoga_quantile_01, racoga_min, racoga_max))
df_racoga = pd.DataFrame(vec_racoga,columns=labels,index = ["mean", "median", "inf-decile","quantile 0.01", "min", "max"])
plt.xlabel("Nb gradient evaluations",fontsize = 13)
plt.ylabel(r"$\log(f)$",fontsize= 13)
plt.legend()
plt.title("Algorithms convergence, N = " + str(N) + ", d = " + str(d))
plt.savefig(path_figure_cv)
plt.figure(figsize=(10,5))
plt.subplot(221)
plt.yticks((racoga["gd"].min(),0,racoga["gd"].max()))
plt.plot(racoga["gd"],label = "GD",color="b")
plt.plot(racoga["nag"],label = "NAG",color="r")
plt.title("RACOGA condition along iterations",fontsize = 10)
plt.legend()
plt.subplot(222)
plt.hist(racoga[index[1]],bins=np.linspace(racoga[index[1]].min(),racoga[index[1]].max(),50),edgecolor="black",facecolor = col,label = labels[1],density = True)

for j in range(nb_rho):
    col = (0.5, j/nb_rho ,1-j/nb_rho)
    plt.hist(racoga[index[j+3]],bins=np.linspace(racoga[index[j+3]].min(),racoga[index[j+3]].max(),50),edgecolor="black",facecolor = col,label = labels[j+3],density = True,alpha = 1-(j)/nb_rho)
plt.title("RACOGA condition number repartition for SNAG",fontsize=10)
plt.legend()
plt.subplot(223)
# plt.plot(racoga[index[1]],label = labels[1],color=col)
min_racoga, max_racoga = racoga[index[3]].min(), racoga[index[3]].max()

for j in range(nb_rho):
    col = (0.5, j/nb_rho ,1-j/nb_rho)
    plt.plot(racoga[index[j+3]],label = labels[j+3],color=col)
    min_racoga, max_racoga = min(min_racoga, racoga[index[j+3]].min()), max(max_racoga, racoga[index[j+3]].max())
plt.title("RACOGA condition number along iterations of SNAG",fontsize=10)
plt.yticks((min_racoga,0,max_racoga))
plt.legend()
plt.savefig(path_figure_racoga)
fig, axs = plt.subplots(figsize = (20,3))
table = axs.table(cellText = df_racoga.values,colLabels=df_racoga.columns,loc="center",rowLabels=df_racoga.index)
axs.axis('off')
table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(13)
axs.set_title("Table of statistics of racoga value along iterations",fontsize=13)
# plt.tight_layout()
plt.savefig(path_figure_racoga + 'table.png')