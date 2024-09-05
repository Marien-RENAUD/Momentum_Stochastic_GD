
for lr in 0.09 0.11 0.13
do
    for momentum in 0.65 0.7 0.75 0.8 0.85
    do
        python train_classifier.py --n_epoch 10 --alg "SNAG" --lr $lr --momentum $momentum --device 1
    done
done