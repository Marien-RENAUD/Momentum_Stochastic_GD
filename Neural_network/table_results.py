# import pandas as pd 
import numpy as np
import torch

device = torch.device('cuda:'+str(1) if torch.cuda.is_available() else 'cpu')

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


## Import loss along algorithms iterations
average_test_sgd = 0
average_test_snag = 0
average_test_nag = 0
average_test_gd = 0
average_test_adam = 0
average_test_rms = 0
for j in range(len(list_seed)):
    dict_results_sgd = torch.load(root + list_net[i] + setting  + "SGD" + list_param[0] + list_seed[j] + "_dict_results.pth")
    dict_results_snag = torch.load(root + list_net[i] + setting  + "SNAG" + list_param[1] + list_seed[j] + "_dict_results.pth")
    dict_results_nag = torch.load(root + list_net[i] + setting  + "NAG" + list_param[2] +list_seed[j] +"_dict_results.pth")
    dict_results_gd = torch.load(root + list_net[i] + setting  + "GD" + list_param[3] +list_seed[j] +"_dict_results.pth")
    dict_results_adam = torch.load(root + list_net[i] + setting  + "ADAM" + list_param[4] +list_seed[j] +"_dict_results.pth")
    dict_results_rms = torch.load(root + list_net[i] + setting  + "RMSprop" + list_param[5] +list_seed[j] +"_dict_results.pth")
    average_test_sgd += torch.tensor(dict_results_sgd["test_accuracy"])
    average_test_snag += torch.tensor(dict_results_snag["test_accuracy"])
    average_test_nag += torch.tensor(dict_results_nag["test_accuracy"])
    average_test_gd += torch.tensor(dict_results_gd["test_accuracy"])
    average_test_adam += torch.tensor(dict_results_adam["test_accuracy"])
    average_test_rms += torch.tensor(dict_results_rms["test_accuracy"])

print(average_test_adam/nb_seed)