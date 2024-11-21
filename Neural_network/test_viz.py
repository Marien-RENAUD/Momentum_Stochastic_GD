import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as t
import os
import matplotlib.pyplot as plt
import numpy as np
from time import time
from tqdm import tqdm
from models_architecture import create_mlp, create_cnn, create_cnn_bn
from argparse import ArgumentParser
import time as time

start = time.time()
parser = ArgumentParser()
parser.add_argument('--data', type=str, default = "CIFAR10", choices = ["CIFAR10", "MNIST", "FashionMNIST", "KMNIST", "EMNIST"])
parser.add_argument('--network_type', type=str, default = "CNN", choices=["CNN", "MLP", "Logistic"])
parser.add_argument('--n_data', type = int, default = None)

hparams = parser.parse_args()
data_choice = hparams.data
network_type = hparams.network_type
n_data = hparams.n_data

def log_ech(x):
    return np.log(1+x/5)

if data_choice == "MNIST":
    path_results_sgd = "results/MNIST/SGD/lr_0.1/CNN_n_epoch_3_batch_random_with_rpl_alg_SGD_lr_0.1_momentum_0.0"
    path_results_snag = "results/MNIST/SNAG/lr_0.05_momentum_0.8/CNN_n_epoch_3_batch_random_with_rpl_alg_SNAG_lr_0.05_momentum_0.8"
   
    list_seed=['0','1','2','3','4','5','6','7','8','9']
    nb_seed = len(list_seed)
    size_vec = len(torch.load(path_results_sgd+ "_seed_" + str(list_seed[0]) +"_dict_loss.pth")["loss_trajectory"])

    #-- Import loss along algorithms iterations --#
    loss_trajectory_sgd = torch.zeros(size_vec)
    loss_trajectory_snag = torch.zeros(size_vec)
    for j in range(len(list_seed)):
        dict_results_sgd = torch.load(path_results_sgd+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        dict_results_snag = torch.load(path_results_snag+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
        loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
    loss_trajectory_sgd /= nb_seed
    loss_trajectory_snag /= nb_seed

    #-- Import RACOGA values --#
    size_vec = len(np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
    racoga_trajectory_sgd = np.zeros(size_vec)
    racoga_trajectory_snag = np.zeros(size_vec)

    dict_racoga_sgd = np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(path_results_snag+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    racoga_trajectory_sgd = np.array(dict_racoga_sgd.item()["racoga_list"])
    racoga_trajectory_snag += np.array(dict_racoga_snag.item()["racoga_list"])

    #-- Plots --#
    # Plot loss
    plt.figure(figsize=(10,5))
    label_size = 20
    legend_size = 17
    number_size = 15
    labelpad_y = -5
    labelpad_x = 2

    plt.subplot(121)
    nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
    col = (0.3, 0.6, 1 - 1*0.8/3)
    plt.plot(np.log(loss_trajectory_sgd),label = "SGD",c = "black",lw=2.5)
    plt.plot(np.log(loss_trajectory_snag),label = "SNAG",c = col,lw=2.5)

    plt.xticks((0, 2500),["0", "2500"], fontsize = number_size)
    plt.yticks((-4,1), ["-4","1"], fontsize = number_size)
    plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad_x)
    plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad_y)

    # Plot RACOGA
    plt.subplot(122)
    nb_gd_eval_det = torch.arange(0,782*5+1  ,782)*0.01
    m = np.min(np.array([(racoga_trajectory_sgd).min(), (racoga_trajectory_snag).min()]))
    m = round(m,1)

    plt.hist(log_ech(racoga_trajectory_sgd),bins="sqrt",edgecolor=None,facecolor = "black",density = True,alpha = 0.5,label = "SGD")
    plt.hist(log_ech(racoga_trajectory_snag),bins="sqrt",edgecolor=None,facecolor = col,density = True,alpha = 0.5,label = "SNAG")

    plt.xticks((log_ech(0), log_ech(m), log_ech(200)),["0", str(m),"200"], fontsize = number_size)
    plt.yticks((0,0.8), ["0","0.8"], fontsize = number_size)
    plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
    # plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
    plt.legend(fontsize = legend_size,frameon=False)

    plt.savefig("figures/alg_cv_" + data_choice + "_10_seeds.png")







if data_choice == "FashionMNIST":
    path_results_sgd = "results/FashionMNIST/SGD/lr_0.25/CNN_n_epoch_3_batch_random_with_rpl_alg_SGD_lr_0.25_momentum_0.0"
    path_results_snag = "results/FashionMNIST/SNAG/lr_0.1_momentum_0.8/CNN_n_epoch_3_batch_random_with_rpl_alg_SNAG_lr_0.1_momentum_0.8"
   
    list_seed=['0','1','2','3','4','5','6','7','8','9']
    nb_seed = len(list_seed)
    size_vec = len(torch.load(path_results_sgd+ "_seed_" + str(list_seed[0]) +"_dict_loss.pth")["loss_trajectory"])

    #-- Import loss along algorithms iterations --#
    loss_trajectory_sgd = torch.zeros(size_vec)
    loss_trajectory_snag = torch.zeros(size_vec)
    for j in range(len(list_seed)):
        dict_results_sgd = torch.load(path_results_sgd+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        dict_results_snag = torch.load(path_results_snag+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
        loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
    loss_trajectory_sgd /= nb_seed
    loss_trajectory_snag /= nb_seed

    #-- Import RACOGA values --#
    size_vec = len(np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
    racoga_trajectory_sgd = np.zeros(size_vec)
    racoga_trajectory_snag = np.zeros(size_vec)

    dict_racoga_sgd = np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(path_results_snag+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    racoga_trajectory_sgd = np.array(dict_racoga_sgd.item()["racoga_list"])
    racoga_trajectory_snag += np.array(dict_racoga_snag.item()["racoga_list"])

    #-- Plots --#
    # Plot loss
    plt.figure(figsize=(10,5))
    label_size = 20
    legend_size = 17
    number_size = 15
    labelpad_y = -5
    labelpad_x = 2

    plt.subplot(121)
    nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
    col = (0.3, 0.6, 1 - 1*0.8/3)
    plt.plot(np.log(loss_trajectory_sgd),label = "SGD",c = "black",lw=2.5)
    plt.plot(np.log(loss_trajectory_snag),label = "SNAG",c = col,lw=2.5)

    plt.xticks((0, 2500),["0", "2500"], fontsize = number_size)
    plt.yticks((-1.5,0.5), ["-1.5","0.5"], fontsize = number_size)
    plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad_x)
    plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad_y)

    # Plot RACOGA
    plt.subplot(122)
    nb_gd_eval_det = torch.arange(0,782*5+1  ,782)*0.01
    m = np.min(np.array([(racoga_trajectory_sgd).min(), (racoga_trajectory_snag).min()]))
    m = round(m,1)

    plt.hist(log_ech(racoga_trajectory_sgd),bins="sqrt",edgecolor=None,facecolor = "black",density = True,alpha = 0.5,label = "SGD")
    plt.hist(log_ech(racoga_trajectory_snag),bins="sqrt",edgecolor=None,facecolor = col,density = True,alpha = 0.5,label = "SNAG")

    plt.xticks((log_ech(0), log_ech(m), log_ech(200)),["0", str(m),"200"], fontsize = number_size)
    plt.yticks((0,1), ["0","1"], fontsize = number_size)
    plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
    # plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
    plt.legend(fontsize = legend_size,frameon=False)

    plt.savefig("figures/alg_cv_" + data_choice + "_10_seeds.png")





if data_choice == "KMNIST":
    path_results_sgd = "results/KMNIST/SGD/lr_0.25/CNN_n_epoch_3_batch_random_with_rpl_alg_SGD_lr_0.25_momentum_0.0"
    path_results_snag = "results/KMNIST/SNAG/lr_0.1_momentum_0.9/CNN_n_epoch_3_batch_random_with_rpl_alg_SNAG_lr_0.1_momentum_0.9"
   
    list_seed=['0','1','2','3','4','5','6','7','8','9']
    nb_seed = len(list_seed)
    size_vec = len(torch.load(path_results_sgd+ "_seed_" + str(list_seed[0]) +"_dict_loss.pth")["loss_trajectory"])

    #-- Import loss along algorithms iterations --#
    loss_trajectory_sgd = torch.zeros(size_vec)
    loss_trajectory_snag = torch.zeros(size_vec)
    for j in range(len(list_seed)):
        dict_results_sgd = torch.load(path_results_sgd+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        dict_results_snag = torch.load(path_results_snag+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
        loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
    loss_trajectory_sgd /= nb_seed
    loss_trajectory_snag /= nb_seed

    #-- Import RACOGA values --#
    size_vec = len(np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
    racoga_trajectory_sgd = np.zeros(size_vec)
    racoga_trajectory_snag = np.zeros(size_vec)

    dict_racoga_sgd = np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(path_results_snag+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    racoga_trajectory_sgd = np.array(dict_racoga_sgd.item()["racoga_list"])
    racoga_trajectory_snag += np.array(dict_racoga_snag.item()["racoga_list"])

    #-- Plots --#
    # Plot loss
    plt.figure(figsize=(10,5))
    label_size = 20
    legend_size = 17
    number_size = 15
    labelpad_y = -5
    labelpad_x = 2

    plt.subplot(121)
    nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
    col = (0.3, 0.6, 1 - 1*0.8/3)
    plt.plot(np.log(loss_trajectory_sgd),label = "SGD",c = "black",lw=2.5)
    plt.plot(np.log(loss_trajectory_snag),label = "SNAG",c = col,lw=2.5)

    plt.xticks((0, 2500),["0", "2500"], fontsize = number_size)
    plt.yticks((-4,1), ["-4","1"], fontsize = number_size)
    plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad_x)
    plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad_y)

    # Plot RACOGA
    plt.subplot(122)
    nb_gd_eval_det = torch.arange(0,782*5+1  ,782)*0.01
    m = np.min(np.array([(racoga_trajectory_sgd).min(), (racoga_trajectory_snag).min()]))
    m = round(m,1)

    plt.hist(log_ech(racoga_trajectory_sgd),bins="sqrt",edgecolor=None,facecolor = "black",density = True,alpha = 0.5,label = "SGD")
    plt.hist(log_ech(racoga_trajectory_snag),bins="sqrt",edgecolor=None,facecolor = col,density = True,alpha = 0.5,label = "SNAG")

    plt.xticks((log_ech(0), log_ech(m), log_ech(200)),["0", str(m),"200"], fontsize = number_size)
    plt.yticks((0,0.8), ["0","0.8"], fontsize = number_size)
    plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
    # plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
    plt.legend(fontsize = legend_size,frameon=False)

    plt.savefig("figures/alg_cv_" + data_choice + "_10_seeds.png")







if data_choice == "EMNIST":
    path_results_sgd = "results/EMNIST/SGD/lr_0.1/CNN_n_epoch_3_batch_random_with_rpl_alg_SGD_lr_0.1_momentum_0.0"
    path_results_snag = "results/EMNIST/SNAG/lr_0.05_momentum_0.9/CNN_n_epoch_3_batch_random_with_rpl_alg_SNAG_lr_0.05_momentum_0.9"
   
    list_seed=['0','1','2','3','4','5','6','7','8','9']
    nb_seed = len(list_seed)
    size_vec = len(torch.load(path_results_sgd+ "_seed_" + str(list_seed[0]) +"_dict_loss.pth")["loss_trajectory"])

    #-- Import loss along algorithms iterations --#
    loss_trajectory_sgd = torch.zeros(size_vec)
    loss_trajectory_snag = torch.zeros(size_vec)
    for j in range(len(list_seed)):
        dict_results_sgd = torch.load(path_results_sgd+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        dict_results_snag = torch.load(path_results_snag+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
        loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
    loss_trajectory_sgd /= nb_seed
    loss_trajectory_snag /= nb_seed

    #-- Import RACOGA values --#
    size_vec = len(np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
    racoga_trajectory_sgd = np.zeros(size_vec)
    racoga_trajectory_snag = np.zeros(size_vec)

    dict_racoga_sgd = np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(path_results_snag+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    racoga_trajectory_sgd = np.array(dict_racoga_sgd.item()["racoga_list"])
    racoga_trajectory_snag += np.array(dict_racoga_snag.item()["racoga_list"])

    #-- Plots --#
    # Plot loss
    plt.figure(figsize=(10,5))
    label_size = 20
    legend_size = 17
    number_size = 15
    labelpad_y = -5
    labelpad_x = 2

    plt.subplot(121)
    nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
    col = (0.3, 0.6, 1 - 1*0.8/3)
    plt.plot(np.log(loss_trajectory_sgd),label = "SGD",c = "black",lw=2.5)
    plt.plot(np.log(loss_trajectory_snag),label = "SNAG",c = col,lw=2.5)

    plt.xticks((0, 2500),["0", "2500"], fontsize = number_size)
    plt.yticks((-5,1), ["-5","1"], fontsize = number_size)
    plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad_x)
    plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad_y)

    # Plot RACOGA
    plt.subplot(122)
    nb_gd_eval_det = torch.arange(0,782*5+1  ,782)*0.01
    m = np.min(np.array([(racoga_trajectory_sgd).min(), (racoga_trajectory_snag).min()]))
    m = round(m,1)

    plt.hist(log_ech(racoga_trajectory_sgd),bins="sqrt",edgecolor=None,facecolor = "black",density = True,alpha = 0.5,label = "SGD")
    plt.hist(log_ech(racoga_trajectory_snag),bins="sqrt",edgecolor=None,facecolor = col,density = True,alpha = 0.5,label = "SNAG")

    plt.xticks((log_ech(0), log_ech(m), log_ech(200)),["0", str(m),"200"], fontsize = number_size)
    plt.yticks((0,2), ["0","2"], fontsize = number_size)
    plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
    # plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
    plt.legend(fontsize = legend_size,frameon=False)

    plt.savefig("figures/alg_cv_" + data_choice + "_10_seeds.png")




if data_choice == "CIFAR10" and network_type == "Logistic":
    path_results_sgd = "results/CIFAR10/SGD/lr_0.25/Logistic_n_epoch_5_batch_random_with_rpl_alg_SGD_lr_0.25_momentum_0.0"
    path_results_snag = "results/CIFAR10/SNAG/lr_0.1_momentum_0.8/Logistic_n_epoch_5_batch_random_with_rpl_alg_SNAG_lr_0.1_momentum_0.8"
   
    list_seed=['0','1','2','3','4','5','6','7','8','9']
    nb_seed = len(list_seed)
    size_vec = len(torch.load(path_results_sgd+ "_seed_" + str(list_seed[0]) +"_dict_loss.pth")["loss_trajectory"])

    #-- Import loss along algorithms iterations --#
    loss_trajectory_sgd = torch.zeros(size_vec)
    loss_trajectory_snag = torch.zeros(size_vec)
    for j in range(len(list_seed)):
        dict_results_sgd = torch.load(path_results_sgd+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        dict_results_snag = torch.load(path_results_snag+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
        loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
    loss_trajectory_sgd /= nb_seed
    loss_trajectory_snag /= nb_seed

    #-- Import RACOGA values --#
    size_vec = len(np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
    racoga_trajectory_sgd = np.zeros(size_vec)
    racoga_trajectory_snag = np.zeros(size_vec)

    dict_racoga_sgd = np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(path_results_snag+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    racoga_trajectory_sgd = np.array(dict_racoga_sgd.item()["racoga_list"])
    racoga_trajectory_snag += np.array(dict_racoga_snag.item()["racoga_list"])

    #-- Plots --#
    # Plot loss
    plt.figure(figsize=(10,5))
    label_size = 20
    legend_size = 17
    number_size = 15
    labelpad_y = -5
    labelpad_x = 2

    plt.subplot(121)
    nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
    col = (0.3, 0.6, 1 - 1*0.8/3)
    plt.plot(np.log(loss_trajectory_sgd),label = "SGD",c = "black",lw=2.5)
    plt.plot(np.log(loss_trajectory_snag),label = "SNAG",c = col,lw=2.5)

    plt.xticks((0, 2500),["0", "2500"], fontsize = number_size)
    plt.yticks((0.65,0.8), ["0.65","0.8"], fontsize = number_size)
    plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad_x)
    plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad_y)

    # Plot RACOGA
    plt.subplot(122)
    nb_gd_eval_det = torch.arange(0,782*5+1  ,782)*0.01
    m = np.min(np.array([(racoga_trajectory_sgd).min(), (racoga_trajectory_snag).min()]))
    m = round(m,1)

    plt.hist(log_ech(racoga_trajectory_sgd),bins="sqrt",edgecolor=None,facecolor = "black",density = True,alpha = 0.5,label = "SGD")
    plt.hist(log_ech(racoga_trajectory_snag),bins="sqrt",edgecolor=None,facecolor = col,density = True,alpha = 0.5,label = "SNAG")

    plt.xticks((log_ech(0), log_ech(m), log_ech(200)),["0", str(m),"200"], fontsize = number_size)
    plt.yticks((0,1.4), ["0","1.4"], fontsize = number_size)
    plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
    # plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
    plt.legend(fontsize = legend_size,frameon=False)

    plt.savefig("figures/alg_cv_" + data_choice + "_"+ network_type +"10_seeds.png")



if data_choice == "CIFAR10" and network_type == "Logistic" and n_data == 10000:
    path_results_sgd = "results/CIFAR10/SGD/lr_1.0/n_data_10000/Logistic_n_epoch_5_batch_random_with_rpl_alg_SGD_lr_1.0_momentum_0.0"
    path_results_snag = "results/CIFAR10/SNAG/lr_0.2_momentum_0.7/n_data_10000/Logistic_n_epoch_5_batch_random_with_rpl_alg_SNAG_lr_0.2_momentum_0.7"
   
    list_seed=['0','1','2','3','4','5','6','7','8','9']
    nb_seed = len(list_seed)
    size_vec = len(torch.load(path_results_sgd+ "_seed_" + str(list_seed[0]) +"_dict_loss.pth")["loss_trajectory"])

    #-- Import loss along algorithms iterations --#
    loss_trajectory_sgd = torch.zeros(size_vec)
    loss_trajectory_snag = torch.zeros(size_vec)
    for j in range(len(list_seed)):
        dict_results_sgd = torch.load(path_results_sgd+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        dict_results_snag = torch.load(path_results_snag+ "_seed_" + str(list_seed[j]) +"_dict_loss.pth")
        loss_trajectory_sgd += torch.tensor(dict_results_sgd["loss_trajectory"])
        loss_trajectory_snag += torch.tensor(dict_results_snag["loss_trajectory"])
    loss_trajectory_sgd /= nb_seed
    loss_trajectory_snag /= nb_seed

    #-- Import RACOGA values --#
    size_vec = len(np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True).item()["racoga_list"])
    racoga_trajectory_sgd = np.zeros(size_vec)
    racoga_trajectory_snag = np.zeros(size_vec)

    dict_racoga_sgd = np.load(path_results_sgd+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(path_results_snag+ "_seed_" + str(list_seed[0]) + "_racoga_results.npy", allow_pickle=True)
    racoga_trajectory_sgd = np.array(dict_racoga_sgd.item()["racoga_list"])
    racoga_trajectory_snag += np.array(dict_racoga_snag.item()["racoga_list"])

    #-- Plots --#
    # Plot loss
    plt.figure(figsize=(10,5))
    label_size = 20
    legend_size = 16
    number_size = 15
    labelpad_y = -5
    labelpad_x = 2

    plt.subplot(121)
    nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
    col = (0.3, 0.6, 1 - 1*0.8/3)
    plt.plot(np.log(loss_trajectory_sgd),label = "SGD",c = "black",lw=2.5)
    plt.plot(np.log(loss_trajectory_snag),label = "SNAG",c = col,lw=2.5)

    plt.xticks((0, 800),["0", "800"], fontsize = number_size)
    plt.yticks((0.65,0.8), ["0.65","0.8"], fontsize = number_size)
    plt.xlabel("Gradient evaluations",fontsize = label_size, labelpad = labelpad_x)
    plt.ylabel(r"$\log(f)$",fontsize = label_size, labelpad = labelpad_y)

    # Plot RACOGA
    plt.subplot(122)
    nb_gd_eval_det = torch.arange(0,782*5+1  ,782)*0.01
    m = np.min(np.array([(racoga_trajectory_sgd).min(), (racoga_trajectory_snag).min()]))
    m = round(m,1)

    plt.hist(log_ech(racoga_trajectory_sgd),bins="sqrt",edgecolor=None,facecolor = "black",density = True,alpha = 0.5,label = "SGD")
    plt.hist(log_ech(racoga_trajectory_snag),bins="sqrt",edgecolor=None,facecolor = col,density = True,alpha = 0.5,label = "SNAG")

    plt.xticks((log_ech(0), log_ech(m), log_ech(20)),["0", str(m),"20"], fontsize = number_size)
    plt.yticks((0,1.6), ["0","1.6"], fontsize = number_size)
    plt.xlabel("RACOGA",fontsize = label_size, labelpad = labelpad_x)
    # plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
    plt.legend(fontsize = legend_size, frameon=False)

    plt.savefig("figures/alg_cv_" + data_choice + "_n_data"+ str(n_data) + network_type +"10_seeds.png")
