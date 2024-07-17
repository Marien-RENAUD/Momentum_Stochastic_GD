# Momentum_Stochastic_GD

This code analyse the converge of optimization framework in a convex and non-convex context. 

## File Structure
```
- dataset : collection of used datasets (CIFAR-10)
- Linear_Regression : code for solving the linear regression optimization problem (with interpolation)
  _ results : directory of experimental results
  _ linear_regression.py : definition of algorithms (GD, NAG, SGD, SNAG)
  _ results.py : code to run algorithms and store functional and racoga values along iterations
  _ vizualisaiton.py : code to plot convergence curve of the algorithms and to track racoga values along iterations
- Neural_Network : code for training Neural Network and compute a convergence analyse
  - results : directory of experimental results
  - models_architecture.py : definition of the CIFAR-10 classifier architecture (MLP or CNN)
  - racoga_computation.py : code for computed the RACOGA quantity for a training
  - train_classifier_cifar10.py : code for training a classifier on CIFAR-10
```
