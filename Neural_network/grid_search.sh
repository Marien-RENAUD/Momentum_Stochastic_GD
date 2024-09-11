
for lr in 0.01 0.05 0.1 0.15 0.2 0.3
do
    for momentum in 0.6 0.7 0.8 0.9
    do
        python train_classifier.py --alg "SNAG" --lr $lr --momentum $momentum --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10
    done
done

for lr in 0.01 0.05 0.1 0.15 0.2 0.3 0.4
do
    python train_classifier.py --alg "SGD" --lr $lr --momentum 0 --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10

done

for lr in 0.5 1 1.5 2 3 4 5 6
do
    python train_classifier.py --alg "GD" --lr $lr --momentum 0 --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10

done


for lr in 0.5 1 2 3 4
do
    for momentum in 0.6 0.7 0.8 0.9
    do
        python train_classifier.py --alg "NAG" --lr $lr --momentum $momentum --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10
    done
done
python train_classifier.py --alg "SNAG" --lr 0.01 --momentum 0.8 --device 1 --data "SPHERE" --grid_search True --network_type "MLP" --n_epoch 10
