# for seed in 38 39 40 41 42
# do
#     python racoga_computation.py --lr 0.05 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "CNN" --data "CIFAR10"
#     python racoga_computation.py --lr 0.3 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10"
#     python racoga_computation.py --lr 2 --alg "NAG" --device 1 --momentum 0.7 --seed $seed --network_type "CNN" --data "CIFAR10"
#     python racoga_computation.py --lr 4 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "CNN" --data "CIFAR10"
# done

for seed in 38 39 40 41 42
do
    python train_classifier.py --lr 0.1 --alg "SNAG" --device 1 --momentum 0.9 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 10
    python train_classifier.py --lr 0.4 --alg "SGD" --device 1 --momentum 0 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 10
    python train_classifier.py --lr 0.5 --alg "NAG" --device 1 --momentum 0.9 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 10
    python train_classifier.py --lr 1 --alg "GD" --device 1 --momentum 0 --seed $seed --network_type "MLP" --data "SPHERE" --n_epoch 10
done