import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

root = 'results/'
setting = "_n_epoch_5_batch_random_with_rpl_alg_"
list_net = ['CNN', 'MLp']
list_param = ['_lr_0.3_momentum_0.0_seed_','_lr_0.05_momentum_0.9_seed_','_lr_2.0_momentum_0.7_seed_','_lr_4.0_momentum_0.0_seed_']
list_seed=['38', '39', '40','41','42']
nb_seed = len(list_seed)
size_vec = len(torch.load(root + list_net[0] + setting  + "SGD" + list_param[0] + list_seed[0] + "_dict_loss.pth")["loss_trajectory"])
size_vec_det = len(torch.load(root + list_net[0] + setting  + "GD" + list_param[3] + list_seed[0] + "_dict_loss.pth")["loss_trajectory"])

i = 0 ## choice of network type : 0 for CNN, 1 for MLP

## Import loss along algorithms iterations
loss_trajectory_sgd = torch.zeros(size_vec)
loss_trajectory_snag = torch.zeros(size_vec)
loss_trajectory_gd = torch.zeros(size_vec_det)
loss_trajectory_nag = torch.zeros(size_vec_det)
for j in range(len(list_seed)):
    dict_results_sgd = torch.load(root + list_net[i] + setting  + "SGD" + list_param[0] + list_seed[j] + "_dict_loss.pth")
    dict_results_snag = torch.load(root + list_net[i] + setting  + "SNAG" + list_param[1] + list_seed[j] + "_dict_loss.pth")
    dict_results_nag = torch.load(root + list_net[i] + setting  + "NAG" + list_param[2] +list_seed[j] +"_dict_loss.pth")
    dict_results_gd = torch.load(root + list_net[i] + setting  + "GD" + list_param[3] +list_seed[j] +"_dict_loss.pth")
    loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
    loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
    loss_trajectory_nag += torch.tensor(dict_results_nag["loss_trajectory"])
    loss_trajectory_gd += torch.tensor(dict_results_gd["loss_trajectory"])
loss_trajectory_sgd /= nb_seed
loss_trajectory_snag /= nb_seed
loss_trajectory_nag /= nb_seed
loss_trajectory_gd /= nb_seed



## Import RACOGA values
size_vec = len(np.load(root + list_net[0] + setting  + "SGD" + list_param[0] + list_seed[0] + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
size_vec_det = len(np.load(root + list_net[0] + setting  + "GD" + list_param[3] + list_seed[0] + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
racoga_trajectory_sgd = np.zeros(size_vec)
racoga_trajectory_snag = np.zeros(size_vec)
racoga_trajectory_gd = np.zeros(size_vec_det)
racoga_trajectory_nag = np.zeros(size_vec_det)
for j in range(len(list_seed)):
    dict_racoga_sgd = np.load(root + list_net[i] + setting  + "SGD" + list_param[0] + list_seed[j] + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(root + list_net[i] + setting  + "SNAG" + list_param[1] + list_seed[j] + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_nag = np.load(root + list_net[i] + setting  + "NAG" + list_param[2] +list_seed[j] +"_racoga_results.npy", allow_pickle=True)
    dict_racoga_gd = np.load(root + list_net[i] + setting  + "GD" + list_param[3] +list_seed[j] +"_racoga_results.npy", allow_pickle=True)
    racoga_trajectory_sgd += np.array(dict_racoga_sgd.item()["racoga_list"])
    racoga_trajectory_snag += np.array(dict_racoga_snag.item()["racoga_list"])
    racoga_trajectory_nag += np.array(dict_racoga_nag.item()["racoga_list"])
    racoga_trajectory_gd += np.array(dict_racoga_gd.item()["racoga_list"])
racoga_trajectory_sgd /= nb_seed
racoga_trajectory_snag /= nb_seed
racoga_trajectory_nag /= nb_seed
racoga_trajectory_gd /= nb_seed


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
labelpad = 2
plt.subplot(121)
nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
plt.plot(loss_trajectory_sgd,label = "SGD",c = "black")
plt.plot(loss_trajectory_snag,label = "SNAG",c = "blue")
plt.plot(nb_gd_eval_det,loss_trajectory_nag,label = "NAG",c = "red",lw=3)
plt.plot(nb_gd_eval_det,loss_trajectory_gd,label = "GD",c = "purple",lw=3,linestyle="--")

plt.xticks((0,782*5 +1), fontsize = number_size)
plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad)
plt.yticks((np.array(loss_trajectory_snag,dtype = np.float32).min(),np.array(loss_trajectory_snag,dtype = np.float32).max()), fontsize = number_size)
plt.legend(fontsize = legend_size)
plt.subplot(122)

nb_gd_eval_det = torch.arange(0,782*5  ,782)*0.01
# plt.plot(racoga_trajectory_sgd,label = "SGD",c = "black")
# plt.plot(racoga_trajectory_snag,label = "SNAG",c = "blue")
# plt.plot(nb_gd_eval_det,racoga_trajectory_gd,label = "GD",c = "purple")
# plt.plot(nb_gd_eval_det,racoga_trajectory_nag,label = "NAG",c = "red")
# plt.xticks((0,80))
# plt.yticks((20,180))

plt.hist(racoga_trajectory_sgd,bins="sqrt",edgecolor=None,facecolor = "black",density = True,alpha = 0.5,label = "SGD")
plt.hist(racoga_trajectory_snag,bins="sqrt",edgecolor=None,facecolor = "blue",density = True,alpha = 0.5,label = "SNAG")
plt.hist(racoga_trajectory_gd,bins="sqrt",edgecolor=None,facecolor = "purple",density = True,alpha = 0.5,label = "GD")
plt.hist(racoga_trajectory_nag,bins="sqrt",edgecolor=None,facecolor = "red",density = True,alpha = 0.5,label = "NAG")
plt.xticks((-1,150), fontsize = number_size)
plt.yticks((0,0.05), fontsize = number_size)
plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad)
# plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
plt.legend(fontsize = legend_size)

plt.savefig("figures/alg_cv_" + list_net[i] + ".png")


