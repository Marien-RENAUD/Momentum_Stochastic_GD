
# for lr in 0.01 0.05 0.1 0.15 0.2 0.3
# do
#     for momentum in 0.6 0.7 0.8 0.9
#     do
#         python train_classifier.py --alg "SNAG" --lr $lr --momentum $momentum --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 5
#     done
# done

# for lr in 0.01 0.05 0.1 0.15 0.2 0.3 0.4
# do
#     python train_classifier.py --alg "SGD" --lr $lr --momentum 0 --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10

# done

# for lr in 0.5 1 1.5 2 3 4 5 6
# do
#     python train_classifier.py --alg "GD" --lr $lr --momentum 0 --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10

# done


# for lr in 0.5 1 2 3 4
# do
#     for momentum in 0.6 0.7 0.8 0.9
#     do
#         python train_classifier.py --alg "NAG" --lr $lr --momentum $momentum --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10
#     done
# done
# python train_classifier.py --alg "SNAG" --lr 0.01 --momentum 0.8 --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10


# for lr in 0.001 0.005 0.01 0.015 0.02
# do
#     for momentum in 0.6 0.7 0.8 0.9
#     do
#         for beta_adam in 0.8 0.9 0.99 0.999
#         do
#             python train_classifier.py --alg "ADAM" --lr $lr --momentum $momentum --beta_adam $beta_adam --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 5
#         done
#     done
# done

# for lr in 0.005 0.01 0.05 0.1 0.2
# do
#     for alpha in 0.8 0.9 0.99 0.999
#     do
#         python train_classifier.py --alg "RMSprop" --lr $lr --alpha_rms $alpha --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 5
#     done
# done

# for lr in 0.005 0.01 0.05 0.1 0.2
# do
#     for alpha in 0.8 0.9 0.99 0.999
#     do
#         python train_classifier.py --alg "RMSprop" --lr $lr --alpha_rms $alpha --device 1 --data "CIFAR10" --grid_search True --network_type "CNN" --n_epoch 5
#     done
# done

# for lr in 0.0001 
# do
#     for alpha in 0.8 0.9 0.99 0.999
#     do
#         python train_classifier.py --alg "RMSprop" --lr $lr --alpha_rms $alpha --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 5
#     done
# done

# for lr in 0.0005 0.001 
# do
#     for alpha in 0.8 0.9 0.99 0.999
#     do
#         python train_classifier.py --alg "RMSprop" --lr $lr --alpha_rms $alpha --device 1 --data "CIFAR10" --grid_search True --network_type "CNN" --n_epoch 5
#     done
# done


# for lr in 0.01 0.05 0.1 0.2 0.5
# # for lr in 0.4
# # for lr in 0.5 0.6
# do
#     for momentum in 0.7 0.8 0.9 0.99
#     do
#         python train_classifier.py --alg "SNAG" --lr $lr --momentum $momentum --device 0 --data "CIFAR10" --network_type "Logistic" --n_epoch 5 --n_data 10000
#     done
# done

for i in 0 1 2 3 4 5 6 7 8 9
do
    python train_classifier.py --alg "SNAG" --lr 0.2 --momentum 0.7 --device 0 --data "CIFAR10" --network_type "Logistic" --n_epoch 5 --seed $i --n_data 10000
done


# for lr in 0.005 0.01 0.05 0.1 0.25 0.5 1
# do
#     python train_classifier.py --alg "SGD" --lr $lr --momentum 0 --device 1 --data "CIFAR10" --network_type "Logistic" --n_epoch 5 --n_data 10000
# done


# for i in 0 1 2 3 4 5 6 7 8 9
# do
#     python train_classifier.py --alg "SGD" --lr 1.0 --momentum 0 --device 1 --data "CIFAR10" --network_type "Logistic" --n_epoch 5 --seed $i --n_data 10000
# done