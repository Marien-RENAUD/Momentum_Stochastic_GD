
for lr in 0.01 0.05 0.1 0.15
do
    for momentum in 0.6 0.7 0.8 0.9
    do
        python train_classifier.py --alg "SNAG" --lr $lr --momentum $momentum --device 1 --data "SPHERE" --grid_search True
    done
done