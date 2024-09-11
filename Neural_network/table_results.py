import pandas as pd 
import numpy as np
import torch

device = torch.device('cuda:'+str(1) if torch.cuda.is_available() else 'cpu')

# path_result_gd = "results/CNN_n_epoch_5_batch_random_with_rpl_alg_GD_lr_4.0_momentum_0.0_seed_"
# path_result_nag = "results/CNN_n_epoch_5_batch_random_with_rpl_alg_NAG_lr_2.0_momentum_0.7_seed_"
# path_result_sgd = "results/CNN_n_epoch_5_batch_random_with_rpl_alg_SGD_lr_0.3_momentum_0.0_seed_"
# path_result_snag = "results/CNN_n_epoch_5_batch_random_with_rpl_alg_SNAG_lr_0.05_momentum_0.9_seed_"
# suffix = '_dict_results.pth'
# seed_list = ['38','39','40','41','42']
# columns = ["GD", "NAG", "SGD","SNAG"]
# rows = ["lr", "momentum" , "test accuracy"]
# lr = ["4", "2", "0.3", "0.05"]
# momentum = ["0", "0.7", "0" , "0.9"]
# acc_gd, acc_sgd, acc_nag, acc_snag = 0,0,0,0
# print("preboucle")
# nb_seed = len(seed_list)
# for i in range(nb_seed):
#     dict_gd = torch.load(path_result_gd + str(seed_list[i]) + suffix)
#     dict_sgd = torch.load(path_result_sgd + str(seed_list[i]) + suffix)
#     dict_nag = torch.load(path_result_nag + str(seed_list[i]) + suffix)
#     dict_snag = torch.load(path_result_snag + str(seed_list[i]) + suffix)
#     acc_gd += dict_gd["test_accuracy"]
#     acc_sgd += dict_sgd["test_accuracy"]
#     acc_nag += dict_nag["test_accuracy"]
#     acc_snag += dict_snag["test_accuracy"]
#     print(i)
# acc_gd /= nb_seed
# acc_sgd /= nb_seed
# acc_nag /= nb_seed
# acc_snag /= nb_seed
# list_acc = [acc_gd,acc_nag,acc_sgd,acc_snag]
# data = [lr,momentum,list_acc]
# print(data)
# df = pd.DataFrame(data,columns=columns,index=rows)
# print(df)

path_result_gd = "results/MLP_n_epoch_10_batch_random_with_rpl_alg_GD_lr_3.0_momentum_0.0_seed_"
path_result_nag = "results/MLP_n_epoch_10_batch_random_with_rpl_alg_NAG_lr_2.0_momentum_0.9_seed_"
path_result_sgd = "results/MLP_n_epoch_10_batch_random_with_rpl_alg_SGD_lr_0.3_momentum_0.0_seed_"
path_result_snag = "results/MLP_n_epoch_10_batch_random_with_rpl_alg_SNAG_lr_0.1_momentum_0.9_seed_"
suffix = '_dict_results.pth'
seed_list = ['38','39','40','41','42']
columns = ["GD", "NAG", "SGD","SNAG"]
rows = ["lr", "momentum" , "test accuracy"]
lr = ["4", "2", "0.3", "0.05"]
momentum = ["0", "0.7", "0" , "0.9"]
acc_gd, acc_sgd, acc_nag, acc_snag = 0,0,0,0
print("preboucle")
nb_seed = len(seed_list)
for i in range(nb_seed):
    dict_gd = torch.load(path_result_gd + str(seed_list[i]) + suffix)
    dict_sgd = torch.load(path_result_sgd + str(seed_list[i]) + suffix)
    dict_nag = torch.load(path_result_nag + str(seed_list[i]) + suffix)
    dict_snag = torch.load(path_result_snag + str(seed_list[i]) + suffix)
    acc_gd += dict_gd["test_accuracy"]
    acc_sgd += dict_sgd["test_accuracy"]
    acc_nag += dict_nag["test_accuracy"]
    acc_snag += dict_snag["test_accuracy"]
    print(i)
acc_gd /= nb_seed
acc_sgd /= nb_seed
acc_nag /= nb_seed
acc_snag /= nb_seed
list_acc = [acc_gd,acc_nag,acc_sgd,acc_snag]
data = [lr,momentum,list_acc]
print(data)
df = pd.DataFrame(data,columns=columns,index=rows)
print(df)