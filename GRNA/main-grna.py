import configparser
import logging
import sys
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import torch.autograd as autograd
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import torchvision.models as tvmodels
from datetime import datetime
from pathlib import Path

from models.GlobalClassifiers import GlobalPreModel_LR, GlobalPreModel_NN
from models.AttackModels import Generator
from utils.Trainers import GlobalClassifierTrainer, GeneratorTrainer
from utils.VFLDatasets import ExperimentDataset, FakeDataset

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dm %ds' % (m, s) if h==0 else '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
def currentDir():
    return os.path.dirname(os.path.realpath(__file__))
    
def parentDir(mydir):
    return str(Path(mydir).parent.absolute())
    
def initlogging(logfile):
    # debug, info, warning, error, critical
    # set up logging to file
    logging.shutdown()
    
    logger = logging.getLogger()
    logger.handlers = []
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=logfile,
                        filemode='w')
    
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.CRITICAL)
    # add formatter to ch
    ch.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(ch)  
    
def getSplittedVarPredictionDataset(trainpart, testpart, predictpart, expset):
    # trainpart + testpart = 0.5, predictpart is variable
    assert parameters['trainpart'] + parameters['testpart'] == 0.5, 'Train + Test should be 0.5'
    x, y=expset[0]
    logging.critical("\n[FUNCTION]: Splitting dataset by getSplittedVarPredictionDataset()......")
    logging.info("Display first (x, y) pair of dataset:\n %s, %s", x, y)
    logging.info("Shape of (x, y): %s %s", x.shape, y.shape)
 
    train_len = int(len(expset) * trainpart)
    test_len = int(len(expset) * testpart)
    predict_len = int(len(expset) * predictpart)
    total_len = int(len(expset))
    
    # make sure (train_len + test_len) == int(total_len/2)
    if int(total_len/2) > (train_len + test_len):
        train_len = train_len + (int(total_len/2) - (train_len + test_len))
    
    remain_len = total_len - train_len - test_len - predict_len
    
    first, second= torch.utils.data.random_split(expset, [int(total_len/2), total_len - int(total_len/2)])
    trainset, testset= torch.utils.data.random_split(first, [train_len, test_len])
    predictset, _= torch.utils.data.random_split(second, [predict_len, remain_len])
    
    logging.critical("len(trainset): %d", len(trainset))
    logging.critical("len(testset): %d", len(testset))
    logging.critical("len(predictset): %d", len(predictset))
    return trainset, testset, predictset
    
def getSplittedDataset(trainpart, testpart, predictpart, expset):
    assert parameters['trainpart'] + parameters['testpart'] + parameters['predictpart'] == 1, 'Train + Test + Prediction should be 1'
    x, y=expset[0]
    logging.critical("\n[FUNCTION]: Splitting dataset by getSplittedDataset()......")
    logging.info("Display first (x, y) pair of dataset:\n %s, %s", x, y)
    logging.info("Shape of (x, y): %s %s", x.shape, y.shape)
 
    train_len = int(len(expset) * trainpart)
    test_len = int(len(expset) * testpart)
    total_len = int(len(expset))
  
    trainset, remainset = torch.utils.data.random_split(expset, [train_len, total_len-train_len])
    logging.critical("len(trainset): %d", len(trainset))

    testset, predictset = torch.utils.data.random_split(remainset, [test_len, len(remainset)-test_len])
    logging.critical("len(testset): %d", len(testset))
    logging.critical("len(predictset): %d", len(predictset))
    return trainset, testset, predictset

def getTimeStamp():
    return datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
 
def readConfigFile(configfile):
    parameters = {}
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(configfile)

    p_default = config['DEFAULT']
    p_dataset = config['DATASET']
    p_defence = config['DEFENCE']
    p_generator = config['GENERATOR']
    p_classifier = config['CLASSIFIER']

    parameters['trainpart'] = p_dataset.getfloat('TrainPortion')
    parameters['testpart'] = p_dataset.getfloat('TestPortion')
    parameters['predictpart'] = p_dataset.getfloat('PredictPortion')
    
    parameters['datasetpath'] = parentDir(currentDir()) + os.sep + "datasets" + os.sep + p_default['DataFile']

    # add time stamp to the name of log file
    logfile = p_default['LogFile']
    index = logfile.rfind('.')
    if index != -1:
        logfile = logfile[:index] + getTimeStamp() + logfile[index:]
    else:
        logfile = logfile + getTimeStamp()
        
    parameters['logpath'] = currentDir() + os.sep + "log" + os.sep + logfile
   
    parameters['n_attacker'] = p_default.getint('NumOfFeaturesOwnedByAttacker')
    parameters['n_victim'] = p_default.getint('NumOfFeaturesToRecover') 
    parameters['runtimes'] = p_default.getint('RunningTimes')
    
    parameters['epochsForAttack'] = p_generator.getint('Epochs')
    parameters['enableAttackerFeatures'] = p_generator.getboolean('EnableAttackerFeatures')
    parameters['enableMean'] = p_generator.getboolean('EnableKnownMeanConstraint')
    parameters['meanLambda'] = p_generator.getfloat('KnownMeanLambda') if parameters['enableMean'] else 0
    parameters['unknownVarLambda'] = p_generator.getfloat('UnknownVarLambda') 
    
    parameters['enableConfRound'] = p_defence.getboolean('EnableConfidenceRounding')
    parameters['roundPrecision'] = p_defence.getint('RoundingPrecision')
    
    parameters['modelType'] = p_classifier['ModeType']
    parameters['outputDim'] = p_classifier.getint('ClassNum')
    parameters['epochsForClassifier'] = p_classifier.getint('Epochs') 

    return parameters
    
def gridSearch(parameters, search_time = 4):
    logging.critical("\n[FUNCTION]: Grid Search %d times......", search_time)
    logging.disable(logging.CRITICAL)  # disable all logging calls with levels <= CRITICAL
    lambda2loss = {}
    max_var_coe = 0.5
    negative_step = 0.005
    positive_step = 0.05
    min_var_coe = -0.03 if parameters['modelType'] != "RF" else 0
    g_input_dim = parameters['n_attacker'] + parameters['n_victim']
    g_output_dim = parameters['n_victim']
    
    for i in range(search_time):
        curr_var = min_var_coe
        # create dataset 
        expset = ExperimentDataset(parameters['datasetpath'])
        x, _ = expset[0]
        assert g_input_dim == x.size(0), "n_attacker + n_victim should be equal to number of features {}".format(x.size(0))
        
        # split dataset and create dataloader
        trainset, testset, predictset = getSplittedDataset(parameters['trainpart'], parameters['testpart'], parameters['predictpart'], expset) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        predictloader = torch.utils.data.DataLoader(predictset, batch_size=64, shuffle=True)

        classifierTrainer = GlobalClassifierTrainer(parameters['modelType'], g_input_dim, parameters['outputDim'], device)
        
        # train classifier and generator
        classifierTrainer.train(trainset, testset, trainloader, testloader, 10)
        if parameters['modelType'] == "RF":
            fakedata = FakeDataset(len(expset)*10, g_input_dim, classifierTrainer.modelRF.rf)
            classifierTrainer.imitateRF(fakedata, epochs=5)
        while curr_var < max_var_coe:
            generatorTrainer = GeneratorTrainer(g_input_dim, g_output_dim, parameters, device)
            generatorTrainer.train(classifierTrainer, predictloader, expset.mean_attr)
            mean_model_loss, _ = generatorTrainer.test(predictloader, expset.mean_attr)
            if lambda2loss.get(curr_var) is None:
                lambda2loss[curr_var] = mean_model_loss
            else:
                lambda2loss[curr_var] = lambda2loss[curr_var] + mean_model_loss
            print("Search time", i+1, "/", search_time, "Searching curr_var", curr_var, "model loss", mean_model_loss)
            if curr_var >0:
                curr_var += positive_step
            else:
                curr_var += negative_step
            
    # sort by loss, ascending order
    lambda2loss = {k: v for k, v in sorted(lambda2loss.items(), key=lambda item: item[1])}
    # enable logging
    logging.disable(logging.NOTSET)
    logging.critical("Grid search result: %s", lambda2loss)
    # get lambda with lowest loss
    final_lambda = next(iter(lambda2loss))
    logging.critical("Choose variance lambda = %s", final_lambda)
    return final_lambda
    
### This attack method is as follows:
#   1. Split the dataset into three parts for training, testing, and prediction
#   2. Train a classification model (logistic regression, random forest or neural network) using train and test data
#   3. Train a generator based on the trained classifier in Step 2 and the prediction dataset
#   4. compute overall mse
    
if __name__=='__main__':  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read parameters from config file
    configfile = 'config.ini'
    parameters = readConfigFile(configfile)
    original_log_path = parameters['logpath']
    
    manualseed = 47
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)
   
    # revise log path
    index = original_log_path.rfind('.')
    formatString = "-attacker-{}-victim-{}-mean-{}-type-{}-predict-{}".format(parameters['n_attacker'], parameters['n_victim'], parameters['enableMean'], parameters['modelType'], parameters['predictpart'])
    if index != -1:
        parameters['logpath'] = original_log_path[:index] + formatString + original_log_path[index:]
    else:
        parameters['logpath'] = original_log_path + formatString
    
    # init logging
    initlogging(parameters['logpath'])
    
    # logging.info("This should be only in file") 
    # logging.critical("This shoud be in both file and console")
    logging.critical("\n\n\n<<<<<<<-------------------------------NumOfFeaturesToRecover: {}------------------------------->>>>>>>".format(parameters['n_victim']))
    logging.critical("Running on device: %s", device)
    logging.critical("Writing log to file: %s", parameters['logpath'])
    logging.critical("n_attacker = %d, n_victim = %d", parameters['n_attacker'], parameters['n_victim'])

    # log config.ini content
    with open(configfile, 'r') as conf: 
        logging.info(conf.read())
        
    g_input_dim = parameters['n_attacker'] + parameters['n_victim']
    g_output_dim = parameters['n_victim']
        
    model_loss = []
    random_loss = []
    runningtime = parameters['runtimes']
    #parameters['unknownVarLambda'] = gridSearch(parameters)
    
    start = time.time()
    
    for count in range(runningtime):
        logging.critical("\n\n<----------------- Running count: %d / %d ----------------->\n\n", count + 1, runningtime)
        # create dataset 
        expset = ExperimentDataset(parameters['datasetpath'])
        logging.critical("For dataset %s, dataset length: %d", parameters['datasetpath'], len(expset))
        x, _ = expset[0]
        assert g_input_dim == x.size(0), "n_attacker + n_victim should be equal to number of features {}".format(x.size(0))
        
        # split dataset and create dataloader
        #trainset, testset, predictset = getSplittedVarPredictionDataset(parameters['trainpart'], parameters['testpart'], parameters['predictpart'], expset) 
        trainset, testset, predictset = getSplittedDataset(parameters['trainpart'], parameters['testpart'], parameters['predictpart'], expset) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
        predictloader = torch.utils.data.DataLoader(predictset, batch_size=64, shuffle=True)

        #rf_predictloader = torch.utils.data.DataLoader(predictset, batch_size=1, shuffle=True) # for computing cbr
        logging.info("len(trainloader): %d", len(trainloader))
        logging.info("len(testloader): %d", len(testloader))
        logging.info("len(predictloader): %d", len(predictloader))
        
        classifierTrainer = GlobalClassifierTrainer(parameters['modelType'], g_input_dim, parameters['outputDim'], device)
        generatorTrainer = GeneratorTrainer(g_input_dim, g_output_dim, parameters, device)
        
        #trees_internal_node_features = None    # for computing cbr
        #trees_internal_node_thresholds = None                                
        # train classifier and generator
        
        classifierTrainer.train(trainset, testset, trainloader, testloader, parameters['epochsForClassifier'])
        if parameters['modelType'] == "RF":
            #trees_internal_node_features, trees_internal_node_thresholds = classifierTrainer.convertRF()   
            fakedata = FakeDataset(len(expset)*10, g_input_dim, classifierTrainer.modelRF.rf)
            classifierTrainer.imitateRF(fakedata, epochs=5)

        generatorTrainer.train(classifierTrainer, predictloader, expset.mean_attr)
        mean_model_loss, mean_guess_loss = generatorTrainer.test(predictloader, expset.mean_attr)
        #mean_model_loss, mean_guess_loss = generatorTrainer.test_rf(rf_predictloader, expset.mean_attr, trees_internal_node_features, trees_internal_node_thresholds)    # for computing cbr                                                         
        model_loss.append(mean_model_loss)
        random_loss.append(mean_guess_loss)
        
        logging.critical('%s (%d%%)' % (timeSince(start, (count + 1) / runningtime), (count+1) / runningtime * 100))
        
    logging.critical("\n\n<----------------- Running Summary ----------------->\n\n")
    model_loss = sorted(model_loss)
    logging.critical("After run %d times -> ", runningtime)
    logging.critical("Mean generator loss: %s", sum(model_loss)/len(model_loss))
    logging.critical("Mean random guess loss: %s", sum(random_loss)/runningtime)
        
        
