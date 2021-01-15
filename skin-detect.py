import random
import numpy
from scipy.stats import multivariate_normal 
import cv2

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
    pos = 0
    neg = 0
    for data in full_set:
        if random.random() < 0.1:
            if data[3] == 1 and neg < 100:
                test_set.append(data)
                neg += 1
            elif data[3] == 2 and pos < 100:
                test_set.append(data)
                pos += 1
        else:
            train_set.append(data)
    print(f"train set size: {len(train_set)}, test set size: {len(test_set)}")
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
 
def skin_detect(input_path,output_path,w=200,h=150,threshold=0.9999999999999):
    full_set = load_data_set()
    (train_set, test_set) = split_into_training_and_testing(full_set)
    (pos_mean,neg_mean,pos_cov,neg_cov,pos_prior,neg_prior) = train(train_set)
    pos_dist = multivariate_normal(mean=pos_mean, cov=pos_cov)
    neg_dist = multivariate_normal(mean=neg_mean, cov=neg_cov)
    
    im = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im,(w,h))
    for x in range(len(im)):
        for y in range(len(im[x])):
            data = im[x][y]    
            pos_cond = pos_dist.pdf(data[0:3])
            neg_cond = neg_dist.pdf(data[0:3])
            skinness = pos_cond * pos_prior / (pos_prior * pos_cond + neg_prior * neg_cond)
            if(skinness < threshold):
                im[x][y] = [0,0,0] 

    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path,im)
    
skin_detect("people.jfif","people-th5.png",threshold=1,w=300,h=300)
skin_detect("people.jfif","people-th0.png",threshold=0,w=300,h=300)
skin_detect("people.jfif","people-th1.png",threshold=0.999999,w=300,h=300)
skin_detect("people.jfif","people-th2.png",threshold=0.999999999999,w=300,h=300)
skin_detect("people.jfif","people-th3.png",threshold=0.9999999999999,w=300,h=300)
skin_detect("people.jfif","people-th4.png",threshold=0.99999999999999,w=300,h=300)
