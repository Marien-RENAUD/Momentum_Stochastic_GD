for seed in 38 39 40 41 42
do
    python convexity_measure.py --lr 0.05 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
    python convexity_measure.py --lr 0.3 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
    python convexity_measure.py --lr 2 --alg "NAG" --device 1 --momentum 0.7 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
    python convexity_measure.py --lr 4 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10" --n_epoch 5
    done
