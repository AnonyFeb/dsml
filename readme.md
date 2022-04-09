## The code is the implementation of paper 'Discrete Scale-invariant Metric Learning for Efficient Collaborative Filtering' submitted to ACM SIGKDD 2022.

## Environment

Python 3.7.9

tensorflow-gpu==1.15.0

numpy==1.19.2

cvxopt

cvxpy

tqdm==4.62.3

pandas==1.3.4

## Prepare Data

1.  Download the following dataset from [Amazon Reviews](http://jmcauley.ucsd.edu/data/amazon/) and [movielens](http://files.grouplens.org/datasets/movielens/), and preprocess datasets with the following steps included in the 'prepareData' directory.

| dataset       |                            |
| ------------- | -------------------------- |
| Books         | 5-core (8,898,041 reviews) |
| Movies and TV | 5-core (1,697,533 reviews) |
| CDs and Vinyl | 5-core (1,097,592 reviews) |
| movielens     | 1-m (1,000,209 reviews)    |

(1).  Run `process.py` to filterout users and items with interaction's number >= 20 

(2).  Run `build_data.py` to create a dictionary to each userID and itemID, **you can change flag to generate two dict files**

(3).  Run `transfrom_data.py` to transfrom string ID to int ID

(4).  Run `trans_id.py` make Implicit data

## Generate Negative Data

1. Run `generate_negitem.py` to generate 5 negative ratings for each user for training. We have prepared MovieLens data in the `Data` directory.

## Train Model

1. Run `dsml.py` to train our model and get the results of 'Ours-real' and 'Ours' methods.


## Saved Model

We have saved hash codes trained with our model as `hash_user_feature.data` and `hash_item_feature.data`
