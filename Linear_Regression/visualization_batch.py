import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os as os
print(os.getcwd() )

version_bis = False # Set true to not overwrite the first experiment
root = "simul_data/batch_" # Data results folder
nb_batch = np.load("nb_batch.npy")
features_type = np.load("features_type.npy") # True : biased features

path_figure_root = 'results/batch/'
if features_type == 0 :
    suffixe = 'unbiased_'
elif features_type == 1 :
    suffixe = 'gaussian_mixture_'
else:
    suffixe = 'alternative_sampling_'

param= torch.load(root +"param_" + suffixe + "batch=" + str(nb_batch) + ".pth")
algo = torch.load(root +"algo_" + suffixe + "batch=" + str(nb_batch) + ".pth")
racoga = torch.load(root +"racoga_" + suffixe + "batch=" + str(nb_batch) + ".pth")
path_figure_cv = path_figure_root + 'convergence/d=' + str(param['d']) + '_N=_' + str(param['N']) + suffixe + '.png'
path_figure_racoga = path_figure_root + "racoga/" + '_d=' + str(param['d']) + '_N=_' + str(param['N']) + suffixe + '.png'

d, N , n_iter, batch_size ,mu , L, L_max = param["d"], param["N"],param["n_iter"], param["batch_size"], param["mu"], param["L"], param["L_max"]
labels = torch.load(root + "labels_" + "batch=" + str(nb_batch) + ".pth")
index = torch.load(root + "index_"+  "batch=" + str(nb_batch) +".pth")

#-- Figures : Racoga convergence (subplot 1) and histograms (subplot 2) --#

plt.figure(figsize=(10,5))
plt.subplot(121)
nb_alg = len(labels) 
nb_rho = len(param["rho"])
nb_gd_eval_det = torch.arange(0,(n_iter+1)*int(batch_size[0]),N)

racoga_mean, racoga_median, racoga_decile_inf, racoga_quantile_01, racoga_min, racoga_max = np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg)
k = 0
ind_batch = 0
for j in range(nb_alg):
    if j==0:
        col = "purple"
    elif j ==1:
        col = "black"
    elif j == 2:
        col = "red"
    else:
        col =  (0.3, 0.6, 1 - 0.8*k/nb_rho )
        k+=1
    if len(algo[index[j]].shape) >1:
        if j == 1:
            nb_gd_eval_sto = torch.arange(0,(n_iter),batch_size[0])
            ind_batch-=1
        else:
            nb_gd_eval_sto = torch.arange(0,int((n_iter+1)*batch_size[0]),int(batch_size[ind_batch]))
        mean_alg,min_alg,max_alg = torch.mean(algo[index[j]],axis=1),torch.min(algo[index[j]],dim=1),torch.max(algo[index[j]],axis=1)    
        plt.plot(nb_gd_eval_sto,torch.log(mean_alg),label=labels[j],color =col,lw=3, alpha = 0.8)
        racoga_current = (racoga[index[j]] + (batch_size[ind_batch]-1)*(N - racoga[index[j]])/(N-1))/batch_size[ind_batch]
        ind_batch += 1

    else:
        plt.plot(nb_gd_eval_det,torch.log(algo[index[j]]),label=labels[j],color =col,lw=2)
        racoga_current = racoga[index[j]]
    racoga_mean[j],racoga_median[j], racoga_decile_inf[j], racoga_quantile_01[j], racoga_min[j], racoga_max[j] = racoga_current.mean(), np.quantile(racoga_current,0.5,method = 'nearest'), np.quantile(racoga_current,0.1,method = 'nearest'), np.quantile(racoga_current,0.01,method = 'nearest'), racoga_current.min(), racoga_current.max()
vec_racoga = np.vstack((racoga_mean, racoga_median, racoga_decile_inf, racoga_quantile_01, racoga_min, racoga_max))
df_racoga = pd.DataFrame(vec_racoga,columns=labels,index = ["mean", "median", "inf-decile","quantile 0.01", "min", "max"])
label_size = 20
legend_size = 10
number_size = 15
labelpad = 2
plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad)
plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad)
plt.xticks((0,n_iter*batch_size[0]), fontsize = number_size)
plt.yticks((12,20), fontsize = number_size)
plt.legend(fontsize = legend_size)
plt.subplot(122)
min_hist = np.nan
max_hist = np.nan
for j in range(nb_rho):
    col = (0.3, 0.6, 1 - 0.8*(j)/nb_rho ) 
    plt.hist(racoga[index[j+3]],bins="sqrt",edgecolor=None,facecolor = col,label = labels[3+j],density = True,alpha = 0.5)
    min_hist,max_hist = np.nanmin([min_hist,racoga[index[3+j]].min()]), np.nanmax([max_hist,racoga[index[3+j]].max()])
plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad)
plt.xticks((-0.5,0,3),["-0.5", "0", "3"], fontsize = number_size)
plt.yticks((0,2), fontsize = number_size)
plt.legend(fontsize = legend_size)
plt.savefig(path_figure_cv)

#-- Figure : Racoga along iterations --#

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
min_racoga, max_racoga = racoga[index[3]].min(), racoga[index[3]].max()
for j in range(nb_rho):
    col = (0.5, j/nb_rho ,1-j/nb_rho)
    racog = (racoga[index[j+3]] + (batch_size[j]-1)*(N - racoga[index[j+3]])/(N-1))/batch_size[j]
    plt.plot(racog,label = labels[j+3],color=col)
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
plt.savefig(path_figure_racoga + 'table.png')
