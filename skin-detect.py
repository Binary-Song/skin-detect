import random
import numpy
from numpy.core.defchararray import translate
from scipy.stats import multivariate_normal

def load_data_set() -> list:
    dataset = []
    with open('./skin_dataset.txt') as dataset_file:
        lines = dataset_file.readlines()
        for line in lines:
            substrs = line.split('\t')
            datapoint = []
            for substr in substrs:
                datapoint.append(int(substr))
            dataset.append(datapoint)
    return numpy.array(dataset)


def split_into_training_and_testing(full_set: list):
    train_set = []
    test_set = [] 
    for data in full_set:
        if random.random() < 0.9:
            test_set.append(data)
        else: 
            train_set.append(data)
    
    return (numpy.array(train_set), numpy.array(test_set))
 

def train(dataset):
    pos_set = dataset[dataset[:,3]==2, 0:3] 
    neg_set = dataset[dataset[:,3]==1, 0:3]
    pos_mean = numpy.mean(pos_set,axis=0)
    neg_mean = numpy.mean(neg_set,axis=0)
    pos_cov = numpy.cov(pos_set,rowvar=False)
    neg_cov = numpy.cov(neg_set,rowvar=False)
    pos_prior = len(pos_set) / len(dataset)
    neg_prior = len(neg_set) / len(dataset)
    return (pos_mean,neg_mean,pos_cov,neg_cov,pos_prior,neg_prior)

full_set = load_data_set()
(train_set, test_set) = split_into_training_and_testing(full_set)
(pos_mean,neg_mean,pos_cov,neg_cov,pos_prior,neg_prior) = train(train_set)
pos_dist = multivariate_normal(mean=pos_mean, cov=pos_cov)
neg_dist = multivariate_normal(mean=neg_mean, cov=neg_cov)

correct = 0
for data in test_set: 
    pos_cond = pos_dist.pdf(data[0:3])
    neg_cond = neg_dist.pdf(data[0:3])
    skinness = pos_cond * pos_prior / (pos_prior * pos_cond + neg_prior * neg_cond)
    if (skinness > 0.5) == (data[3] == 2):
        correct += 1

print(f"correct projection count: {correct} out of {len(test_set)} ({correct/len(test_set)})")
