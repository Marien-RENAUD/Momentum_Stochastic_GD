# for lr in 1 3 5 7
# do
#     python train_classifier.py --n_epoch 10 --lr $lr --alg "GD" --device 1
# done

# for lr in 1 3 5 7
# do
#     for momentum in 0.6 0.7 0.8 0.9
#     do
#         python train_classifier.py --n_epoch 10 --lr $lr --alg "NAG" --device 1 --momentum $momentum
#     done
# done

# for seed in 33 34 35 36 37 38 39 40 41 42
# do
#     python train_classifier.py --lr 0.1 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
#     python train_classifier.py --lr 0.3 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
#     python train_classifier.py --lr 2 --alg "NAG" --device 1 --momentum 0.9 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
#     python train_classifier.py --lr 3 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
# done

# for seed in 33 34 35 36 37 38 39 40 41 42
# do
#     python train_classifier.py --lr 0.05 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python train_classifier.py --lr 0.3 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python train_classifier.py --lr 2 --alg "NAG" --device 1 --momentum 0.7 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python train_classifier.py --lr 4 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
# done

for seed in 33 34 35 36 37 38 39 40 41 42
do
    python train_classifier.py --lr 0.001 --beta_adam 0.8 --alg "ADAM" --device 1 --momentum 0.7 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
     python train_classifier.py --lr 0.005 --beta_adam 0.8 --alg "ADAM" --device 1 --momentum 0.6 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5

done
