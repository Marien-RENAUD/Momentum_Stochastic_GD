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

#python train_classifier.py --lr 1 --alg "NAG" --device 1 --momentum 0.6

#python train_classifier.py --lr 1 --alg "GD" --device 1 --momentum 0
for seed in 40 41 42
do
#    python train_classifier.py --lr 0.05 --alg "SNAG" --device 1 --momentum 0.85 --seed $seed
    python train_classifier.py --lr 0.15 --alg "SGD" --device 1 --momentum 0 --seed $seed
done