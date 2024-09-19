import numpy as np
import torch
import matplotlib.pyplot as plt
# import pandas as pd

include_adam = True ## Set True to also plot ADAM 
include_RMSprop = True ## Set True to also plot RMSprop
root = 'results/'
setting = "_n_epoch_5_batch_random_with_rpl_alg_"
list_net = ['CNN', 'MLP']
i = 1  ## choice of network type : 0 for CNN, 1 for MLP
if i == 0:
    list_param = ['_lr_0.3_momentum_0.0_seed_','_lr_0.05_momentum_0.9_seed_','_lr_2.0_momentum_0.7_seed_','_lr_4.0_momentum_0.0_seed_','_lr_0.005_momentum_0.6_beta_0.8_seed_','_lr_0.005_alpha_0.9_seed_']
if i == 1:
    list_param = ['_lr_0.3_momentum_0.0_seed_','_lr_0.1_momentum_0.9_seed_','_lr_2.0_momentum_0.9_seed_','_lr_3.0_momentum_0.0_seed_','_lr_0.001_momentum_0.7_beta_0.8_seed_','_lr_0.001_alpha_0.8_seed_']
list_seed=['33','34','35','36','37','38', '39', '40','41','42']
nb_seed = len(list_seed)
size_vec = len(torch.load(root + list_net[i] + setting  + "SGD" + list_param[0] + list_seed[0] + "_dict_loss.pth")["loss_trajectory"])
size_vec_det = len(torch.load(root + list_net[i] + setting  + "GD" + list_param[3] + list_seed[0] + "_dict_loss.pth")["loss_trajectory"])



## Import loss along algorithms iterations
loss_trajectory_sgd = torch.zeros(size_vec)
loss_trajectory_snag = torch.zeros(size_vec)
loss_trajectory_nag = torch.zeros(size_vec_det)
loss_trajectory_gd = torch.zeros(size_vec_det)
loss_trajectory_adam = torch.zeros(size_vec)
loss_trajectory_rms = torch.zeros(size_vec)
for j in range(len(list_seed)):
    dict_results_sgd = torch.load(root + list_net[i] + setting  + "SGD" + list_param[0] + list_seed[j] + "_dict_loss.pth")
    dict_results_snag = torch.load(root + list_net[i] + setting  + "SNAG" + list_param[1] + list_seed[j] + "_dict_loss.pth")
    dict_results_nag = torch.load(root + list_net[i] + setting  + "NAG" + list_param[2] +list_seed[j] +"_dict_loss.pth")
    dict_results_gd = torch.load(root + list_net[i] + setting  + "GD" + list_param[3] +list_seed[j] +"_dict_loss.pth")
    dict_results_adam = torch.load(root + list_net[i] + setting  + "ADAM" + list_param[4] +list_seed[j] +"_dict_loss.pth")
    dict_results_rms = torch.load(root + list_net[i] + setting  + "RMSprop" + list_param[5] +list_seed[j] +"_dict_loss.pth")
    loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
    loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
    loss_trajectory_nag += torch.tensor(dict_results_nag["loss_trajectory"])
    loss_trajectory_gd += torch.tensor(dict_results_gd["loss_trajectory"])
    loss_trajectory_adam += torch.tensor(dict_results_adam["loss_trajectory"])
    loss_trajectory_rms += torch.tensor(dict_results_rms["loss_trajectory"])
loss_trajectory_sgd /= nb_seed
loss_trajectory_snag /= nb_seed
loss_trajectory_nag /= nb_seed
loss_trajectory_gd /= nb_seed
loss_trajectory_adam /= nb_seed
loss_trajectory_rms /= nb_seed



## Import RACOGA values
size_vec = len(np.load(root + list_net[i] + setting  + "SGD" + list_param[0] + list_seed[0] + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
size_vec_det = len(np.load(root + list_net[i] + setting  + "GD" + list_param[3] + list_seed[0] + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
racoga_trajectory_sgd = np.zeros(size_vec)
racoga_trajectory_snag = np.zeros(size_vec)
racoga_trajectory_gd = np.zeros(size_vec_det)
racoga_trajectory_nag = np.zeros(size_vec_det)
racoga_trajectory_adam = np.zeros(size_vec)
racoga_trajectory_rms = np.zeros(size_vec)
for j in range(len(list_seed)):
    dict_racoga_sgd = np.load(root + list_net[i] + setting  + "SGD" + list_param[0] + list_seed[j] + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(root + list_net[i] + setting  + "SNAG" + list_param[1] + list_seed[j] + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_nag = np.load(root + list_net[i] + setting  + "NAG" + list_param[2] +list_seed[j] +"_racoga_results.npy", allow_pickle=True)
    dict_racoga_gd = np.load(root + list_net[i] + setting  + "GD" + list_param[3] +list_seed[j] +"_racoga_results.npy", allow_pickle=True)
    dict_racoga_adam = np.load(root + list_net[i] + setting  + "ADAM" + list_param[4] +list_seed[j] +"_racoga_results.npy", allow_pickle=True)
    dict_racoga_rms = np.load(root + list_net[i] + setting  + "RMSprop" + list_param[5] +list_seed[j] +"_racoga_results.npy", allow_pickle=True)
    racoga_trajectory_sgd += np.array(dict_racoga_sgd.item()["racoga_list"])
    racoga_trajectory_snag += np.array(dict_racoga_snag.item()["racoga_list"])
    racoga_trajectory_nag += np.array(dict_racoga_nag.item()["racoga_list"])
    racoga_trajectory_gd += np.array(dict_racoga_gd.item()["racoga_list"])
    racoga_trajectory_adam += np.array(dict_racoga_adam.item()["racoga_list"])
    racoga_trajectory_rms += np.array(dict_racoga_rms.item()["racoga_list"])

racoga_trajectory_sgd /= nb_seed
racoga_trajectory_snag /= nb_seed
racoga_trajectory_nag /= nb_seed
racoga_trajectory_gd /= nb_seed
racoga_trajectory_adam /= nb_seed
racoga_trajectory_rms /= nb_seed


# setting = "_n_epoch_10_batch_random_with_rpl_alg_"
# dict_racoga_sgd = np.load(root + list_net[i] + setting  + "SGD_racoga_results.npy", allow_pickle=True)
# dict_racoga_snag = np.load(root + list_net[i] + setting  + "SNAG_racoga_results.npy", allow_pickle=True)
# dict_racoga_nag = np.load(root + list_net[i] + setting  + "NAG_racoga_results.npy", allow_pickle=True)
# dict_racoga_gd = np.load(root + list_net[i] + setting  + "GD_racoga_results.npy", allow_pickle=True)
# print(type(dict_racoga_gd))
# racoga_trajectory_sgd = dict_racoga_sgd.item()['racoga_list']
# racoga_trajectory_snag = dict_racoga_snag.item()["racoga_list"]
# racoga_trajectory_gd = dict_racoga_gd.item()["racoga_list"]
# racoga_trajectory_nag = dict_racoga_nag.item()["racoga_list"]

plt.figure(figsize=(10,5))
label_size = 20
legend_size = 10
number_size = 15
labelpad_y = -20
labelpad_x = 2
plt.subplot(121)
nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
col = (0.3, 0.6, 1 - 1*0.8/3)
plt.plot(loss_trajectory_sgd[:3911],label = "SGD",c = "black",lw=2.5)
plt.plot(loss_trajectory_snag[:3911],label = "SNAG",c = col,lw=2.5)


plt.plot(nb_gd_eval_det,loss_trajectory_nag[:6],label = "NAG",c = "red",lw=3)
plt.plot(nb_gd_eval_det,loss_trajectory_gd[:6],label = "GD",c = "purple",lw=3,linestyle="--")
if include_adam == True:
    plt.plot(loss_trajectory_adam[:3911],label = "ADAM",c = "yellow",lw=2.5)
if include_RMSprop == True:
    plt.plot(loss_trajectory_rms[:3911],label = "RMSprop",c = "pink",lw=2.5)
plt.xticks((0,782*5 +1), fontsize = number_size)
plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad_x)
plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad_y)
# plt.yticks((0,2.4),["0","2.4"], fontsize = number_size)
# plt.ylim((0,2.4))
plt.yticks((0,0.75),["0","0.75"], fontsize = number_size)
plt.ylim((0,0.75))
plt.xlim((-500,3911))

plt.legend(fontsize = legend_size,frameon=False)#, bbox_to_anchor=(0.45, 0.45))
label_size = 20
legend_size = 15
number_size = 15
labelpad_y = -20
labelpad_x = 2
plt.subplot(122)

nb_gd_eval_det = torch.arange(0,782*5+1  ,782)*0.01
# plt.plot(racoga_trajectory_sgd[:40],label = "SGD",c = "black",lw=3)
# plt.plot(racoga_trajectory_snag[:40],label = "SNAG",c = "blue",lw=3)
# plt.plot(nb_gd_eval_det,racoga_trajectory_gd[:6],label = "GD",c = "purple",lw=3,linestyle="--")
# plt.plot(nb_gd_eval_det,racoga_trajectory_nag[:6],label = "NAG",c = "red",lw=3)
# plt.xticks((0,32))
# plt.yticks((10,150))

plt.hist(racoga_trajectory_sgd[:3911],bins="sqrt",edgecolor=None,facecolor = "black",density = True,alpha = 0.5,label = "SGD")
plt.hist(racoga_trajectory_snag[:3911],bins="sqrt",edgecolor=None,facecolor = col,density = True,alpha = 0.5,label = "SNAG")
plt.hist(racoga_trajectory_gd[:6],bins="sqrt",edgecolor=None,facecolor = "purple",density = True,alpha = 0.5,label = "GD")
plt.hist(racoga_trajectory_nag[:6],bins="sqrt",edgecolor=None,facecolor = "red",density = True,alpha = 0.5,label = "NAG")
if include_adam == True:
    plt.hist(racoga_trajectory_adam[:3911],bins="sqrt",edgecolor=None,facecolor = "yellow",density = True,alpha = 0.5,label = "ADAM")
if include_adam == True:
    plt.hist(racoga_trajectory_rms[:3911],bins="sqrt",edgecolor=None,facecolor = "pink",density = True,alpha = 0.5,label = "RMSprop")
plt.xticks((-0.5,150),["-0.5","150"], fontsize = number_size)
plt.yticks((0,0.06), ["0","0.06"], fontsize = number_size)
plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
# plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
plt.legend(fontsize = legend_size,frameon=False)
if include_adam == True & include_RMSprop == False:
    plt.savefig("figures/alg_cv_" + list_net[i] + "_10_seeds_with_ADAM.png")
elif include_RMSprop == True  & include_adam == False:
    plt.savefig("figures/alg_cv_" + list_net[i] + "_10_seeds_with_RMSprop.png")
elif include_RMSprop == True  & include_adam == True:
    plt.savefig("figures/alg_cv_" + list_net[i] + "_10_seeds_with_RMSprop_and_ADAM.png")
else:
    plt.savefig("figures/alg_cv_" + list_net[i] + "_10_seeds.png")


