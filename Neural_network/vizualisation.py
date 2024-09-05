import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

root = 'results/'
setting = "_n_epoch_5_batch_random_with_rpl_alg_"
list_net = ['MLP','CNN']

for i in range(len(list_net)):
    
    ## Import loss along algorithms iterations
    dict_results_sgd = torch.load(root + list_net[i] + setting  + "SGD_dict_loss.pth")
    dict_results_snag = torch.load(root + list_net[i] + setting  + "SNAG_dict_loss.pth")
    dict_results_nag = torch.load(root + list_net[i] + setting  + "NAG_dict_loss.pth")
    dict_results_gd = torch.load(root + list_net[i] + setting  + "GD_dict_loss.pth")
    loss_trajectory_sgd = dict_results_sgd["loss_trajectory"]
    loss_trajectory_snag = dict_results_snag["loss_trajectory"]
    loss_trajectory_gd = dict_results_gd["loss_trajectory"]
    loss_trajectory_nag = dict_results_nag["loss_trajectory"]

    ## Import RACOGA values
    dict_racoga_sgd = np.load(root + list_net[i] + setting  + "SGD_racoga_results.npy", allow_pickle=True)
    dict_racoga_snag = np.load(root + list_net[i] + setting  + "SNAG_racoga_results.npy", allow_pickle=True)
    dict_racoga_nag = np.load(root + list_net[i] + setting  + "NAG_racoga_results.npy", allow_pickle=True)
    dict_racoga_gd = np.load(root + list_net[i] + setting  + "GD_racoga_results.npy", allow_pickle=True)
    print(type(dict_racoga_gd))
    racoga_trajectory_sgd = dict_racoga_sgd.item()['racoga_list']
    racoga_trajectory_snag = dict_racoga_snag.item()["racoga_list"]
    racoga_trajectory_gd = dict_racoga_gd.item()["racoga_list"]
    racoga_trajectory_nag = dict_racoga_nag.item()["racoga_list"]

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    nb_gd_eval_det = torch.arange(0,782*5 +1 ,782)
    plt.plot(loss_trajectory_sgd,label = "SGD",c = "black")
    plt.plot(loss_trajectory_snag,label = "SNAG",c = "blue")
    plt.plot(nb_gd_eval_det,loss_trajectory_gd,label = "GD",c = "purple")
    plt.plot(nb_gd_eval_det,loss_trajectory_nag,label = "NAG",c = "red")
    plt.xticks((0,782*5 +1))
    plt.yticks((np.array(loss_trajectory_snag,dtype = np.float32).min(),np.array(loss_trajectory_snag,dtype = np.float32).max()))
    plt.legend()
    plt.subplot(122)
    nb_gd_eval_det = torch.linspace(0,(782*5 +1)*0.01 ,782*0.01)
    plt.plot(racoga_trajectory_sgd,label = "SGD",c = "black")
    plt.plot(racoga_trajectory_snag,label = "SNAG",c = "blue")
    plt.plot(nb_gd_eval_det,racoga_trajectory_gd,label = "GD",c = "purple")
    plt.plot(nb_gd_eval_det,racoga_trajectory_nag,label = "NAG",c = "red")
    # plt.xticks((0,782*5 +1))
    # plt.yticks((np.array(racoga_trajectory_snag,dtype = np.float32).min(),np.array(racoga_trajectory_snag,dtype = np.float32).max()))
    plt.legend()

    plt.savefig("figures/alg_cv_" + list_net[i] + ".png")


