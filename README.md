# Deep Learning - Imbalanced Dataset 

__Random under-sampling approach and also a comparison of different optimizers: Adam, SGD, Adadelta, Adagrad, and 
various learning rates.__


__Keywords:__ Deep learning, Imbalanced Dataset, Resampling

## Resampling

A widely adopted technique for dealing with highly unbalanced datasets is called resampling. It consists 
of removing samples from the majority class (under-sampling) and / or adding more examples from the minority 
class (over-sampling).

[image info](docs/resampling.png)

Despite the advantage of balancing classes, these techniques also have their weaknesses (there is no free lunch). 
The simplest implementation of over-sampling is to duplicate random records from the minority class, which can 
cause overfitting. In under-sampling, the simplest technique involves removing random records from the majority 
class, which can cause loss of information.

## Dirs and files

- [Notebook](https://github.com/adions025/imbalanced_dataset-/blob/master/nbs/imbalanced_dataset.ipynb)
- [Source code](https://github.com/adions025/imbalanced_dataset-/tree/master/src)
- [Result file](https://github.com/adions025/imbalanced_dataset-/blob/master/data/results.csv)

## Author

* **Adonis González Godoy** ([Email](adions025@gmail.com) - [Github](https://github.com/adions025))

