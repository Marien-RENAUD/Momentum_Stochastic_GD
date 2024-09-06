import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

root_GD = 'results/MLP_n_epoch_5_batch_random_with_rpl_alg_GD_lr_3.0_momentum_0.0_seed_42_dict_loss.pth'
root_NAG = 'results/MLP_n_epoch_5_batch_random_with_rpl_alg_NAG_lr_2.0_momentum_0.6_seed_42_dict_loss.pth'
root_SGD = 'results/MLP_n_epoch_5_batch_random_with_rpl_alg_SGD_lr_0.15_momentum_0.0_seed_'
root_SNAG = 'results/MLP_n_epoch_5_batch_random_with_rpl_alg_SNAG_lr_0.05_momentum_0.9_seed_'
dict_results_nag = torch.load(root_NAG)
dict_results_gd = torch.load(root_GD)
list_seed=['40','41','42']
size_vec = len(torch.load(root_SGD + list_seed[0] + "_dict_loss.pth")["loss_trajectory"])
loss_trajectory_sgd = torch.zeros(size_vec)
loss_trajectory_snag = torch.zeros(size_vec)  
for i in range(len(list_seed)):
    dict_results_sgd = torch.load(root_SGD + list_seed[i] + "_dict_loss.pth")
    dict_results_snag = torch.load(root_SNAG  +list_seed[i] + "_dict_loss.pth")
    loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
    loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
loss_trajectory_sgd /= 3
loss_trajectory_snag /= 3
loss_trajectory_gd = dict_results_gd["loss_trajectory"]
loss_trajectory_nag = dict_results_nag["loss_trajectory"]
plt.figure(figsize=(10,5))
nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
plt.plot(loss_trajectory_sgd,label = "SGD",c = "black")
plt.plot(loss_trajectory_snag,label = "SNAG",c = "blue")
plt.plot(nb_gd_eval_det,loss_trajectory_gd,label = "GD",c = "purple")
plt.plot(nb_gd_eval_det,loss_trajectory_nag,label = "NAG",c = "red")
plt.xticks((0,782*5 +1))
plt.yticks((np.array(loss_trajectory_snag,dtype = np.float32).min(),np.array(loss_trajectory_snag,dtype = np.float32).max()))
plt.legend()
plt.savefig("figures/alg_cv_MLP_bis" + ".png")