# for seed in 38 39 40 41 42
# do
#     python racoga_computation.py --lr 0.05 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "CNN" --data "CIFAR10"
#     python racoga_computation.py --lr 0.3 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10"
#     python racoga_computation.py --lr 2 --alg "NAG" --device 1 --momentum 0.7 --seed $seed --network_type "CNN" --data "CIFAR10"
#     python racoga_computation.py --lr 4 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10"
# done

# for seed in 33 34 35 36 37 38 39 40 41 42
# do
#     python racoga_computation.py --lr 0.1 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
#     python racoga_computation.py --lr 0.3 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
#     python racoga_computation.py --lr 2 --alg "NAG" --device 1 --momentum 0.9 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
#     python racoga_computation.py --lr 3 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
# done

# for seed in 33 34 35 36 37 38 39 40 41 42
# do
#     python racoga_computation.py --lr 0.05 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python racoga_computation.py --lr 0.3 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python racoga_computation.py --lr 2 --alg "NAG" --device 1 --momentum 0.7 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python racoga_computation.py --lr 4 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
# done

# for seed in 38 39 40 41 42
# do
#     python racoga_computation.py --lr 0.05 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python racoga_computation.py --lr 0.3 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python racoga_computation.py --lr 2 --alg "NAG" --device 1 --momentum 0.7 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     python racoga_computation.py --lr 4 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
#     done


# for seed in 33 34 35 36 37 38 39 40 41 42
# do
#     python racoga_computation.py --lr 0.001 --beta_adam 0.8 --alg "ADAM" --device 1 --momentum 0.7 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
#      python racoga_computation.py --lr 0.005 --beta_adam 0.8 --alg "ADAM" --device 1 --momentum 0.6 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5

# done

# for seed in 33 34 35 36 37 38 39 40 41 42
for seed in 36 37 38 39 40 41 42
do
    python racoga_computation.py --lr 0.001 --alpha_rms 0.8 --alg "RMSprop" --device 1 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 5
    python racoga_computation.py --lr 0.005 --alpha_rms 0.9 --alg "RMSprop" --device 1 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5

done