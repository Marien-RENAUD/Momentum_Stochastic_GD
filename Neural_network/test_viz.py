import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# vec = torch.load("results/MLP_n_epoch_4_batch_random_with_rpl_alg_NAG_lr_1.0_momentum_0.9_seed_42_dict_results.pth")
# racoga = vec["racoga"]
# racoga = torch.tensor(racoga)
# print(racoga)
# plt.plot(racoga)

vec = np.load("results/MLP_n_epoch_10_batch_random_with_rpl_alg_SNAG_lr_0.1_momentum_0.9_seed_42_convexity_test_results.npy",allow_pickle=True)
conv_diff = vec.item()["convexity_diff_list"]
plt.plot(conv_diff)
plt.savefig("figures/convexity_diff.png")