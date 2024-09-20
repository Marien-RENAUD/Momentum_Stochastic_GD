import numpy as np
import torch
import matplotlib.pyplot as plt
# import pandas as pd
import os as os

print(os.getcwd() )

version_bis = False # Set true to not overwrite the first experiment
root = "simul_data/" # Data results folder
nb_rho = np.load("nb_rho.npy")
features_type = np.load("features_type.npy") # True : biased features
# features_type = 0
if features_type == 0 :
    path_figure_root= 'results/unbiased_features/'
    suffixe = 'unbiased_'
else :
    path_figure_root = 'results/gaussian_mixture/'
    suffixe = 'gaussian_mixture_' 
param= torch.load(root +"param_" + suffixe + 'rho='+ str(nb_rho) + ".pth")
algo = torch.load(root +"algo_" + suffixe + 'rho='+ str(nb_rho) + ".pth")
racoga = torch.load(root +"racoga_" + suffixe + 'rho='+ str(nb_rho) + ".pth")
corr_data = torch.load(root +"corr_data_"+ suffixe + 'rho='+ str(nb_rho)+ ".pth")
path_figure_cv = path_figure_root + 'convergence/d=' + str(param['d']) + '_N=_' + str(param['N']) + suffixe + '.png'
path_figure_racoga = path_figure_root + "racoga/" + '_d=' + str(param['d']) + '_N=_' + str(param['N']) + suffixe + '.png'
d, N , n_iter, batch_size ,mu , L, L_max = param["d"], param["N"],param["n_iter"], param["batch_size"], param["mu"], param["L"], param["L_max"]
labels = torch.load(root + "labels_" + suffixe  + "rho="+ str(nb_rho) + ".pth")
index = torch.load(root + "index_"+ suffixe + "rho="+ str(nb_rho) + ".pth")

#-- Figures : Racoga convergence (subplot 1) and histograms (subplot 2) --#

plt.figure(figsize=(10,5))
plt.subplot(121)
nb_alg = len(labels) 
nb_rho = len(param["rho"])
nb_gd_eval_det = torch.arange(0,(n_iter+1)*batch_size,N)
nb_gd_eval_sto = torch.arange(0,(n_iter)*batch_size,batch_size)
racoga_mean, racoga_median, racoga_decile_inf, racoga_quantile_01, racoga_min, racoga_max = np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg), np.empty(nb_alg)
k = 0
for j in range(nb_alg):
    if j==0:
        col = "purple"
    elif j ==1:
        col = "black"
    elif j == 2:
        col = "red"
    else:
        col = (0.3, 0.6, 1 - 0.8*k/nb_rho )
        k+=1
    if len(algo[index[j]].shape) >1:
        mean_alg,min_alg,max_alg = torch.mean(algo[index[j]],axis=1),torch.min(algo[index[j]],dim=1),torch.max(algo[index[j]],axis=1)
        plt.plot(nb_gd_eval_sto,torch.log(mean_alg),label=labels[j],color =col,lw=3, alpha = 0.8)
    else:
        plt.plot(nb_gd_eval_det,torch.log(algo[index[j]]),label=labels[j],color =col,lw=2)
    racoga_current = racoga[index[j]]
    racoga_mean[j],racoga_median[j], racoga_decile_inf[j], racoga_quantile_01[j], racoga_min[j], racoga_max[j] = racoga_current.mean(), np.quantile(racoga_current,0.5,method = 'nearest'), np.quantile(racoga_current,0.1,method = 'nearest'), np.quantile(racoga_current,0.01,method = 'nearest'), racoga_current.min(), racoga_current.max()
vec_racoga = np.vstack((racoga_mean, racoga_median, racoga_decile_inf, racoga_quantile_01, racoga_min, racoga_max))
# df_racoga = pd.DataFrame(vec_racoga,columns=labels,index = ["mean", "median", "inf-decile","quantile 0.01", "min", "max"])
label_size = 20
legend_size = 15
number_size = 15
labelpad_y = -20
labelpad_x = 2
plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad_x)
plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad_y)
plt.xticks((0,n_iter*batch_size), fontsize = number_size)
# plt.yticks((12,19), fontsize = number_size)
plt.yticks((-8,9))
plt.legend(loc = "lower left",fontsize = legend_size,frameon = False, bbox_to_anchor=(-0.01,-0.03))
plt.subplot(122)
min_hist = np.nan
max_hist = np.nan
for j in range(nb_rho):
    col = (0.3, 0.6, 1 - 0.8*(j)/nb_rho )
    plt.hist(racoga[index[3+j]],bins="sqrt",edgecolor=None,facecolor = col,label = labels[3+j],density = True,alpha = 0.5)
    min_hist,max_hist = np.nanmin([min_hist,racoga[index[3+j]].min()]), np.nanmax([max_hist,racoga[index[3+j]].max()])
plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
# plt.xticks((-0.5,0,2.5),["-0.5", "0", "2.5"], fontsize = number_size)
plt.xticks((-0.12,0,0.12),["-0.12", "0", "0.12"], fontsize = number_size)
plt.yticks((0,20), fontsize = number_size)
plt.legend(fontsize = legend_size,frameon = False)
plt.savefig(path_figure_cv)

#-- Figure : Racoga along iterations --#

plt.figure(figsize=(5,5))
legend_size = 10
labelpad_y = -20
labelpad_x = -10
zero_vec = np.zeros(n_iter-N)
nb_gd_eval_det = torch.arange(0,(n_iter)*batch_size,N)
plt.yticks((-0.2,0,0.1), ["-0.2","0","0.1"],fontsize = number_size)
plt.plot(nb_gd_eval_det,racoga["gd"],label = "GD",color="b",lw=3)
plt.plot(nb_gd_eval_det,racoga["nag"],label = "NAG",color="r",lw=3)
min_racoga, max_racoga = racoga[index[3]].min(), racoga[index[3]].max()
for j in range(nb_rho):
    col = (0.5, j/nb_rho ,1-j/nb_rho)
    plt.plot(racoga[index[j+3]][:((n_iter)-(N))],label = labels[j+3],color=col,lw=2,alpha = 0.9)
    min_racoga, max_racoga = min(min_racoga, racoga[index[j+3]].min()), max(max_racoga, racoga[index[j+3]].max())
plt.plot(zero_vec,linestyle = "--", lw = 3, color = "black")
plt.ylabel("RACOGA",fontsize = label_size, labelpad = labelpad_y)
plt.xlabel("Gradient evaluations",fontsize = label_size,labelpad = labelpad_x)
plt.xticks((0,900),fontsize = number_size)
plt.legend(fontsize = legend_size)
plt.savefig(path_figure_racoga)
