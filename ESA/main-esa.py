import sys
import argparse
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torchvision.models as tvmodels
import matplotlib.animation as animation
from IPython.display import HTML
import logging
import time
from numpy import genfromtxt
import configparser
import os 
from pathlib import Path
from sklearn.model_selection import train_test_split

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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

def readConfigFile(configfile):
    parameters = {}
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(configfile)

    
    p_dataset = config['DATASET']
    parameters['train_percentage'] = p_dataset.getfloat('TrainPortion')
    parameters['test_percentage'] = p_dataset.getfloat('TestPortion')
    parameters['pred_percentage'] = p_dataset.getfloat('PredictPortion')
    
    p_default = config['DEFAULT']
    parameters['data_path'] = parentDir(currentDir()) + os.sep + "datasets" + os.sep + p_default['DataFile']

    parameters['num_target_features'] = p_default.getint('NumOfFeaturesToRecover') 
    parameters['num_exps'] = p_default.getint('RunningTimes')
    parameters['epoch_num'] = p_default.getint('Epochs')
    
    # add time stamp to the name of log file
    logfile = p_default['LogFile']
    index = logfile.rfind('.')
    if index != -1:
        logfile = logfile[:index]  + "_" + "unknown_" + str(parameters['num_target_features']) \
                + "_expnum_" + str(parameters['num_exps']) \
                + "_" + time.strftime("%Y%m%d%H%M%S") + logfile[index:]
    else:
        logfile = logfile + "_" + "unknown_" + str(parameters['num_target_features']) \
                + "_expnum_" + str(parameters['num_exps']) \
                + "_" + time.strftime("%Y%m%d%H%M%S") + ".log"
        
    parameters['logpath'] = currentDir() + os.sep + "log" + os.sep + logfile

    return parameters
    
### This attack method is as follows:
#   1. Split the dataset into train, test, and predict;
#   2. Train a model (logistic regression) using train and test data
#   3. For each sample in predict data, do the following:
#       3.1 compute the ground-truth prediction (with confidence values)
#       3.2 determine the target features
#       3.3 random initialize the target feature values
#       3.4 equality-solving attack
#       3.5 compute mse
#   4. compute overall mse
    
if __name__=='__main__':  
    manualseed = 50
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)
    
    # read parameters from config file
    configfile = 'config.ini'
    parameters = readConfigFile(configfile)
    
    epochs = parameters['epoch_num']
    trials = parameters['num_exps']
    
    # init logging
    initlogging(parameters['logpath'])
    # logging.info("This should be only in file") 
    # logging.critical("This shoud be in both file and console")
    
    logging.critical('dataset: %s', parameters['data_path'])
    logging.critical('number of target features: %d', parameters['num_target_features'])
    logging.critical('number of experimental trials: %d', trials)
    logging.critical('Begin trials')
    back_prop_mse, random_guess_mse = AverageMeter(), AverageMeter()

    # Create training, testing and prediction dataset
    total_feature_num = None
    class_labels = None
    for it in range(trials):
        logging.critical('-------------------------------------')
        logging.critical("Attack trial {} / {}".format(it+1, trials))
        class InputDataset(Dataset):
            def __init__(self):
                full_data_table = np.genfromtxt(parameters['data_path'], delimiter=',')
                global total_feature_num, class_labels
                total_feature_num = full_data_table[:, :-1].shape[1]
                class_labels = np.unique(full_data_table[:,-1])
                data = torch.from_numpy(full_data_table).float()
                self.samples = data[:, :-1]
                # permuate columns
                batch, columns = self.samples.size()
                permu_cols = torch.randperm(columns)
                logging.critical("Dataset column permutation is: \n %s", permu_cols)
                self.samples = self.samples[:, permu_cols]

                self.labels = data[:, -1]
                min, _ = self.samples.min(dim=0)
                max, _ = self.samples.max(dim=0)
                self.feature_min = min
                self.feature_max = max
                # print(min.shape, max.shape)

                self.samples = (self.samples - self.feature_min) / (self.feature_max - self.feature_min)
                #print("Len(samples):", len(self.labels), "Positive labels sum:", self.labels.sum().item())

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, index):
                return self.samples[index], self.labels[index]

        dataset = InputDataset()
        logging.info('dataset size: %d', len(dataset))
        class_num = len(class_labels)
        logging.info('total feature num: %d', total_feature_num)
        logging.info('class labels: {}'.format(class_labels))
        logging.info('class num: %d', len(class_labels))

        total_sample_num = len(dataset)
        pred_set_num = int(total_sample_num * parameters['pred_percentage'])
        test_set_num = int((total_sample_num - pred_set_num) * parameters['test_percentage'])
        train_set_num = total_sample_num - pred_set_num - test_set_num

        pred_set, test_set, train_set = torch.utils.data.random_split(dataset, [pred_set_num, test_set_num, train_set_num])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        logging.info('len(train_loader): %d', len(train_loader))
        logging.info('len(test_loader): %d', len(test_loader))

        # create a Multi-class LR model that input layer is size of total_feature_num
        class TargetModel(nn.Module):
            def __init__(self):
                super(TargetModel, self).__init__()
                self.dense = nn.Sequential(
                    nn.Linear(total_feature_num, class_num, False),
                    nn.Softmax()
                )

            def forward(self, x):
                return self.dense(x)

        target_model = TargetModel()
        # print(target_model)
        interval = len(train_loader) - 1
        optimizer = torch.optim.Adam(target_model.parameters())
        criteria = torch.nn.NLLLoss()

        # train model
        def train(model, train_loader, optimizer, epoch):
            model.train()
            batch_idx = 0
            for x, y in train_loader:
                optimizer.zero_grad()
                yhat = model(x)
                #loss = F.nll_loss(yhat, y.long())
                loss = criteria(yhat, y.long())
                loss.backward()
                optimizer.step()
                batch_idx += 1

                if batch_idx % interval == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))

        def test(model, dataloader):
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for x, y in dataloader:
                    yhat = model(x)
                    #test_loss += F.nll_loss(yhat, y.long(), reduction='sum').item()
                    test_loss += criteria(yhat, y.long())
                    pred = yhat.argmax(dim=1, keepdim=True) # get the index of the max probability
                    correct += pred.eq(y.view_as(pred)).sum().item()
                    #logging.info('correct: %d', correct)

            test_loss /= len(dataloader.dataset)
            #logging.info('correct num: %d', correct)
            logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

        scheduler = StepLR(optimizer, step_size=1)
        for epoch in range(1, epochs + 1):
            train(target_model, train_loader, optimizer, epoch)
            test(target_model, test_loader)
            scheduler.step()

        linear_layer_params = None
        # print(linear_layer_params)
        for name, param in target_model.named_parameters():
            if param.requires_grad:
                # print('1')
                # print(name, param.data)
                linear_layer_params = param.data

        # compute the required coefficients for attack
        linear_layer_params_left = linear_layer_params[:class_num-1, :]
        # print('params left: ', linear_layer_params_left)
        linear_layer_params_right = linear_layer_params[1:, :]
        # print('params right: ', linear_layer_params_right)
        linear_layer_params_sub = linear_layer_params_left - linear_layer_params_right
        # print('params sub: ', linear_layer_params_sub)

        params_target = linear_layer_params_sub[:, total_feature_num - parameters['num_target_features']:]
        params_adv = linear_layer_params_sub[:, :total_feature_num - parameters['num_target_features']]
        # print('params target shape: ', params_target.shape)
        # print('params adversary shape: ', params_adv.shape)
        
        # Computes the pseudoinverse (Moore-Penrose inverse) of the params_target matrix
        params_target_inv = torch.pinverse(params_target)
        # print('params target inverse shape: ', params_target_inv.shape)

        def attack(ground_truth, input_sample, id):
            noise = torch.randn(parameters['num_target_features'])
            t = noise.sigmoid()
            random_mse = ((t - input_sample[total_feature_num - parameters['num_target_features']:]) ** 2).mean()
            input_sample.resize_((total_feature_num, 1))
            input_sample_adv = input_sample[:total_feature_num - parameters['num_target_features'], :]
            input_sample_target = input_sample[total_feature_num - parameters['num_target_features']: , :]
            with torch.no_grad():
                ground_truth_ln = torch.log(ground_truth)
            ground_truth_ln_left = ground_truth_ln[:class_num-1]
            ground_truth_ln_right = ground_truth_ln[1:]
            ground_truth_ln_diff = ground_truth_ln_left - ground_truth_ln_right
            ground_truth_ln_diff.resize_(class_num-1, 1)
            a = torch.matmul(params_adv, input_sample_adv)
            b = ground_truth_ln_diff - a
            x_target = torch.matmul(params_target_inv, b)
            attack_mse = ((x_target - input_sample_target) ** 2).mean()
            # if id == 0:
                # print('input_sample_adv: ', input_sample_adv)
                # print('input_sample_adv shape: ', input_sample_adv.shape)
                # print('input_sample_target: ', input_sample_target)
                # print('input_sample_target shape: ', input_sample_target.shape)
                # print('ground_truth_ln_diff: ', ground_truth_ln_diff)
                # print('ground_truth_ln_diff shape: ', ground_truth_ln_diff.shape)
                # print('a shape: ', a.shape)
                # print('b shape: ', b.shape)
                # print('attack_mse: ', attack_mse)
                # print('x_target: ', x_target)
                # print('x_target.shape: ', x_target.shape)
            return attack_mse, random_mse

        total_attack_mse = 0.0
        total_rand_mse = 0.0
        pred_interval = 1000
        for i in range(pred_set_num):
            sample, label = pred_set.__getitem__(i)
            y_ground_truth = target_model(sample)
            # if i == 0:
                # print('sample 0 prediction ground truth: ', y_ground_truth)
                # print(y_ground_truth)
            back_attack_mse, rand_mse = attack(y_ground_truth, sample, i)
            total_attack_mse += back_attack_mse
            total_rand_mse += rand_mse
            if i % pred_interval == 0 and i != 0:
                logging.info('current index is: %d', i)
                logging.info('current attack mse is: %f', total_attack_mse/i)
                logging.info('current random mse is: %f', total_rand_mse/i)
        cur_average_attack_mse = total_attack_mse/pred_set_num
        cur_average_rand_mse = total_rand_mse/pred_set_num
        logging.critical('average attack mse: %f', cur_average_attack_mse)
        logging.critical('average random mse: %f', cur_average_rand_mse)
        back_prop_mse.update(cur_average_attack_mse)
        random_guess_mse.update(cur_average_rand_mse)

    logging.critical('-------------------------------------')
    logging.critical(f'Back propagation attack\t {trials}-Avg MSE:\t{back_prop_mse.avg:4f}')
    logging.critical(f'Random guess attack\t {trials}-Avg MSE:\t{random_guess_mse.avg:4f}')
    print("See {} for more details.".format(parameters['logpath']))