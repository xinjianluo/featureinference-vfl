import configparser
import logging

import torch
import torch.optim as optim
from sklearn.tree import _tree, export_text
from sklearn import tree
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

from models.GlobalClassifiers import GlobalPreModel_LR, GlobalPreModel_NN, GlobalPreModel_RF
from models.AttackModels import Generator, FakeRandomForest

def getTimeStamp():
    return datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    
class GlobalClassifierTrainer():
    def __init__(self, modeltype, input_dim, output_dim, device):
        super().__init__()
        logging.critical("\n[FUNCTION]: Creating GlobalClassifierTrainer......")
        logging.critical("Creating a model for type %s", modeltype)
        self.modeltype = modeltype 
        self.device = device
        if modeltype == "LR":
            #assert output_dim == 2, "Output dimension of Logistic Regression should be 2!"
            self.model = GlobalPreModel_LR(input_dim, output_dim).to(device)
        elif modeltype == "NN":
            self.model = GlobalPreModel_NN(input_dim, output_dim).to(device)
        elif modeltype == "RF":
            self.modelRF = GlobalPreModel_RF(trees=100, depth=3, r_state=0)
            self.model = FakeRandomForest(input_dim, output_dim).to(device)
            logging.info("Structure of Fake Random Forest: %s", self.model)
        
        logging.info("Structure of Global Classifier: %s", self.model if modeltype != "RF" else self.modelRF.rf)
            
    def train(self, trainset, testset, trainloader, testloader, epochs):
        logging.critical("\n[FUNCTION]: Training global classifier %s......", self.modeltype)
        if self.modeltype == "LR":
            self.trainLR(trainloader, testloader, epochs)
        elif self.modeltype == "NN":
            self.trainNN(trainloader, testloader, epochs)
        elif self.modeltype == "RF":
            self.trainRF(trainset, testset)
            
            
    def saveModel(self, filename=""):
        if self.modeltype == "RF":
            logging.critical("CANNOT save RF model!")
            return None 
        # add time stamp to the model name
        index = filename.rfind('.')
        righthalf = ""
        if index != -1:
            righthalf = filename[index:]
            filename = filename[:index]
     
        filename = filename + "-" + self.model.__class__.__name__ + getTimeStamp() + righthalf
        logging.critical("Save model to %s", filename)
        torch.save(self.model.state_dict(), filename)
        
    def loadModel(self, filename):
        assert self.modeltype != "RF", "CANNOT load RF model!"
        logging.critical("Load global classifier from %s", filename)
        self.model.load_state_dict(torch.load(filename)) 
        
    def trainRF(self, trainset, testset):
        def predictAccuracy(rf, x, y):
            # x, y should be tensors
            yhat = rf.predict_proba(x.numpy())
            yhat = torch.from_numpy(yhat)
            base = yhat.size(0)
            count = (yhat.argmax(dim=1) == y).sum()
            return count.item() / base 
 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=True)
        x, y = next(iter(trainloader))
        x_test, y_test = next(iter(testloader))
        self.modelRF.rf.fit(x.numpy(), y.numpy())
        train = predictAccuracy(self.modelRF.rf, x, y)
        test = predictAccuracy(self.modelRF.rf, x_test, y_test)
        logging.critical("After training Random Forest, train accur is {}, test accur is {}.".format(train, test))
        
    def convertRF(self):        
        # for computing cbr, convert the trained decision tree model to list
        num_trees = self.modelRF.rf.n_estimators
        print('num_trees: ', num_trees)
        trees = self.modelRF.rf.estimators_
        max_depth = self.modelRF.rf.max_depth
        print('max_depth: ', max_depth)
        max_node_num = int(math.pow(2, max_depth + 1) - 1)

        # init internal nodes and feature thresholds
        trees_internal_node_features = np.full((num_trees, max_node_num), -1, dtype=int)
        trees_internal_node_thresholds = np.full((num_trees, max_node_num), -1, dtype=float)

        # recursive traverse tree_ and store info in the four arrays
        def tree_recurse(t, i, tree_node, full_node_id, depth):
            tree_ = t.tree_
            if depth <= tree_.max_depth + 1:
                if tree_.feature[tree_node] != _tree.TREE_UNDEFINED:
                    threshold = tree_.threshold[tree_node]
                    feature = tree_.feature[tree_node]
                    trees_internal_node_features[i, full_node_id] = feature
                    trees_internal_node_thresholds[i, full_node_id] = threshold
                    tree_recurse(t, i, tree_.children_left[tree_node], 2 * full_node_id + 1, depth + 1)
                    tree_recurse(t, i, tree_.children_right[tree_node], 2 * full_node_id + 2, depth + 1)

        i = 0
        for t in trees:
            #print('i: ', i)
            #r = export_text(t, show_weights=True)
            #print(r)
            tree_recurse(t, i, 0, 0, 1)
            i = i + 1

        #print(trees_internal_node_features)
        #print(trees_internal_node_thresholds)
        #print(trees_internal_node_features.shape)
        return trees_internal_node_features, trees_internal_node_thresholds    
        
        
    def imitateRF(self, fakedata, epochs=5):
        def check_test_accuracy(model, dataloader, loss_fn):
            ll = []
            model.eval()
            with torch.no_grad():
                for x, y in dataloader: 
                    x = x.to(device)
                    y = y.to(device)
                    yhat = model(x)
                    ll.append(loss_fn(yhat, y))
            return sum(ll)/len(ll)
        
        assert self.modeltype == "RF", "imitateRF() is only appliable to RF"
        logging.critical("\n[FUNCTION]: Imitating Random Forest......")
        knowndata = fakedata
        knowntrainset, knowntestset = torch.utils.data.random_split(knowndata, [len(knowndata) - int(len(knowndata) * 0.2), int(len(knowndata) * 0.2)])
        knowntrainloader = torch.utils.data.DataLoader(knowntrainset, batch_size=64, shuffle=True)
        knowntestloader = torch.utils.data.DataLoader(knowntestset, batch_size=64, shuffle=True)
        
        fakeRFOpt = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.MSELoss()
        test_interval = int(epochs / 5)
        self.model.train()
        device = self.device
        for epoch in range(epochs):
            for x, y in knowntrainloader:
                x = x.to(device)
                y = y.to(device)
                fakeRFOpt.zero_grad()
                yhat = self.model(x)  
                loss = loss_fn(yhat, y)
                loss.backward()
                fakeRFOpt.step() 

            if epoch % test_interval == 0 or epoch == epochs - 1:
                test_loss = check_test_accuracy(self.model, knowntestloader, loss_fn)
                logging.critical("In epoch {}, train loss is {}, test loss is {}.".format(epoch, loss.item(), test_loss)) 
                self.model.train()
            
    def trainNN(self, trainloader, testloader, epochs):
        def check_test_accuracy(mymodel, dataloader):
            mymodel.eval()
            accur = 0.0
            base = 0
            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    yhat = mymodel(x)
                    accur += ( (yhat.argmax(dim=1)) == y ).sum()
                    base += x.shape[0]
            return accur / base
        
        device = self.device
        optimizer = torch.optim.Adam(self.model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        
        test_interval = int(epochs / 5)
        self.model.train()

        for epoch in range(epochs):
            accurate = 0.0
            train_accur_base = 0.0
            for x, y in trainloader: 
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                yhat = self.model(x)
                loss = loss_fn(yhat, y.long())
                loss.backward()
                optimizer.step()
                accurate += ((yhat.argmax(dim=1))==y).sum()  
                train_accur_base += x.shape[0]
                
            if epoch % test_interval == 0 or epoch == epochs - 1:
                # for each epoch, print information
                train = accurate / train_accur_base
                test = check_test_accuracy(self.model, testloader)
                self.model.train()
                logging.critical("In epoch {}, train accur is {}, test accur is {}.".format(epoch, train, test))

    def trainLR(self, trainloader, testloader, epochs):
        def check_test_accuracy(model, dataloader):
            model.eval()
            accur = 0.0
            base = 0 
            with torch.no_grad():
                for x, y in dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    yhat = model(x)
                    #accur += ( (torch.zeros_like(y) + (yhat > 0.5 ).squeeze() ) == y).sum()
                    accur += ( (yhat.argmax(dim=1)) == y ).sum()
                    base += x.shape[0]
            return accur / base
        device = self.device
        #loss_fn = torch.nn.BCELoss()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        test_interval = int(epochs / 5) 
        self.model.train()

        for epoch in range(epochs):
            accurate = 0.0
            train_accur_base = 0.0
            for x, y in trainloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                yhat = self.model(x)  
                loss = loss_fn(yhat, y.long())
                loss.backward()
                optimizer.step()
                #accurate += ( (torch.zeros_like(y) + (yhat > 0.5 ).squeeze() ) == y).sum()
                accurate += ((yhat.argmax(dim=1))==y).sum() 
                train_accur_base += x.shape[0]

            if epoch % test_interval == 0 or epoch == epochs - 1:
                train = accurate / train_accur_base
                test = check_test_accuracy(self.model, testloader)
                self.model.train()
                logging.critical("In epoch {}, train accur is {}, test accur is {}.".format(epoch, train, test))
        
    
class GeneratorTrainer():
    def __init__(self, input_dim, output_dim, parameters, device):
        super().__init__()
        logging.critical("\n[FUNCTION]: Creating GeneratorTrainer......")
        logging.critical("Creating a generator")
        # create generator and classifier
        self.netG = Generator(input_dim, output_dim).to(device)
        logging.info("Structure of Generator: %s", self.netG)  
        self.device = device
        self.parameters = parameters
        
    def train(self, classifierTrainer, predictloader, mean_feature):
        logging.critical("\n[FUNCTION]: Training Generator......")
        netR = classifierTrainer.model
        netR.eval()
        self.netG.train()
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr = 0.0001)

        device = self.device
        parameters = self.parameters
        epochs = parameters['epochsForAttack']
        log_interval = int(epochs / 5) 
        
        n_attacker = parameters['n_attacker']
        n_victim = parameters['n_victim']
            
        enableAttackerFeatures = parameters['enableAttackerFeatures']
        logging.critical('Enable Attacker Features') if enableAttackerFeatures else logging.critical('Disable Attacker Features')
        enableMean = parameters['enableMean']
        logging.critical('Enable Mean Constraint') if enableMean else logging.critical('Disable Mean Constraint')
        logging.critical('parameters[\'unknownVarLambda\'] = %s', parameters['unknownVarLambda']) 
        enableConfRound = parameters['enableConfRound']
        logging.critical('Enable Confidence Rounding') if enableConfRound else logging.critical('Disable Confidence Rounding') 
        logging.critical('parameters[\'roundPrecision\'] = %s', parameters['roundPrecision'])
        
        for epoch in range(epochs):
            accurate = 0.0
            train_accur_base = 0.0
            losses = []
            for x, y in predictloader:
                optimizerG.zero_grad()
                x = x.to(device)
                y = y.to(device)
                if enableAttackerFeatures:
                    noise = torch.randn(x.size(0), n_victim).to(device)
                    fake_input2netG = torch.cat((x[:, :n_attacker], noise), dim=1)
                else:   
                    fake_input2netG = torch.randn(x.size(0), n_attacker + n_victim)
       
                yhat = self.netG(fake_input2netG) 
                ycat = torch.cat((x[:, :n_attacker], yhat), dim=1)
                yfinal = netR(ycat)
                mean_loss = 0 
                unknown_var_loss = 0 
              
                for i in range(yhat.size(1)):
                    if enableMean:                       
                        mean_loss = mean_loss + (yhat[:, i].mean() - mean_feature[n_attacker + i])**2  # # mean() known
                    unknown_var_loss = unknown_var_loss + (yhat[:, i].var())     # var() unknown
                #if parameters['modelType'] == "RF":
                #    ground_truth = classifierTrainer.modelRF.rf.predict_proba(x.numpy())
                #    ground_truth = torch.from_numpy(ground_truth)
                #else:
                ground_truth = netR(x)
                if enableConfRound:
                    n_digits = parameters['roundPrecision']
                    ground_truth = torch.round(ground_truth * 10**n_digits) / (10**n_digits) 
                    
                loss = ((yfinal - ground_truth.detach())**2).sum() + parameters['meanLambda'] * mean_loss + \
                     + parameters['unknownVarLambda'] * unknown_var_loss
                    
                loss.backward()
                losses.append(loss.detach())
                optimizerG.step() 

            if epoch % log_interval == 0 or epoch == epochs - 1:
                logging.info('>>>>>>>>>>>>>>>>>')
                logging.critical("In epoch %d, loss is %s", epoch, sum(losses) / len(losses) ) 
                logging.info("L2 norm of yhat: %s", (yhat**2).sum())
                logging.info("L2 norm of original vector: %s", (x[:, n_attacker:]**2).sum())
                logging.info("First two lines of yhat: %s", yhat[:2, :])

    def test(self, predictloader, mean_feature):
        def lossPerFeature(input, target):
            res = []
            for i in range(input.size(1)):
                loss = ((input[:, i] - target[:, i])**2).mean().item()
                res.append(loss)
            return np.array(res) 
            
        logging.critical("\n[FUNCTION]: Testing Generator......")
        netG = self.netG 
        parameters = self.parameters
        device = self.device
        n_attacker = parameters['n_attacker']
        n_victim = parameters['n_victim']
        
        netG.eval()
        
        mse = torch.nn.MSELoss(reduction='mean')
        generator_losses = []
        random_losses = [] 
        output = 10
        total_model_loss_pf = None
        total_random_loss_pf = None
        enableAttackerFeatures = parameters['enableAttackerFeatures']
        logging.critical('Enable Attacker Features') if enableAttackerFeatures else logging.critical('Disable Attacker Features')
        enableMean = parameters['enableMean']
        logging.critical('Enable Mean Constraint') if enableMean else logging.critical('Disable Mean Constraint')
           
        for x, y in predictloader:
            x = x.to(device)
            y = y.to(device)
            if enableAttackerFeatures:    
                noise = torch.randn(x.size(0), n_victim).to(device)
                fake_input2netG = torch.cat((x[:, :n_attacker], noise), dim=1)
            else:  
                fake_input2netG = torch.randn(x.size(0), n_attacker + n_victim)
             
            yhat = netG(fake_input2netG)

            if enableMean:
                randomguess = mean_feature[n_attacker:].repeat(x.size(0), 1)
                randomguess = randomguess + torch.normal(0, 1/2, size=randomguess.size())
                randomguess = randomguess.clamp(0, 1)
            else:
                randomguess = torch.rand_like(yhat)
                
            randomguess = randomguess.to(device)
                
            model_loss = mse(x[:, n_attacker:], yhat).item()
            rand_loss = mse(x[:, n_attacker:], randomguess).item()
            
            generator_losses.append(model_loss)
            random_losses.append(rand_loss)
            model_loss_pf = lossPerFeature(x[:, n_attacker:], yhat)
            random_loss_pf = lossPerFeature(x[:, n_attacker:], randomguess)
            total_model_loss_pf = model_loss_pf if total_model_loss_pf is None else total_model_loss_pf + model_loss_pf
            total_random_loss_pf = random_loss_pf if total_random_loss_pf is None else total_random_loss_pf + random_loss_pf
            
            if output>0:
                logging.critical("<<<<<<<<<<<<<<<<")
                logging.critical("Model output loss: %s", model_loss)
                logging.critical("Random guess loss: %s", rand_loss)
                logging.info("Model output loss PF: %s", model_loss_pf)
                logging.info("Random guess loss PF: %s", random_loss_pf)
                logging.info("Attack result, first 2 samples: %s", yhat[:2, :])
                logging.info("Ground truth, first 2 samples: %s:", x[:2, n_attacker:])
                output -= 1 
        logging.critical("------------------ SUMMARY ------------------")
        mean_model_loss = sum(generator_losses)/len(generator_losses)
        mean_guess_loss = sum(random_losses)/len(random_losses)
        mean_model_loss_pf = total_model_loss_pf / len(generator_losses)
        mean_guess_loss_pf = total_random_loss_pf / len(random_losses)
        logging.critical("Mean generator loss: %s", mean_model_loss)
        logging.critical("Mean random guess loss: %s", mean_guess_loss)
        logging.critical("Mean generator loss Per Feature: %s", mean_model_loss_pf)
        logging.critical("Mean random guess loss Per Feature: %s", mean_guess_loss_pf)
        return mean_model_loss, mean_guess_loss
        
    def test_rf(self, predictloader, mean_feature, trees_internal_node_features, trees_internal_node_thresholds):
        # for computing cbr
        logging.critical("\n[FUNCTION]: Testing Generator......")
        netG = self.netG
        parameters = self.parameters
        device = self.device
        n_attacker = parameters['n_attacker']
        n_victim = parameters['n_victim']

        netG.eval()

        generator_total_branch_num = 0
        generator_correct_branch_num = 0
        random_total_branch_num = 0
        random_correct_branch_num = 0
        # output = 10
        # total_model_loss_pf = None
        # total_random_loss_pf = None
        enableAttackerFeatures = parameters['enableAttackerFeatures']
        logging.critical('Enable Attacker Features') if enableAttackerFeatures else logging.critical(
            'Disable Attacker Features')
        enableMean = parameters['enableMean']
        logging.critical('Enable Mean Constraint') if enableMean else logging.critical('Disable Mean Constraint')

        for x, y in predictloader:
            x = x.to(device)
            y = y.to(device)
            if enableAttackerFeatures:
                noise = torch.randn(x.size(0), n_victim).to(device)
                fake_input2netG = torch.cat((x[:, :n_attacker], noise), dim=1)
            else:
                fake_input2netG = torch.randn(x.size(0), n_attacker + n_victim)

            yhat = netG(fake_input2netG)

            if enableMean:
                randomguess = mean_feature[n_attacker:].repeat(x.size(0), 1)
                randomguess = randomguess + torch.normal(0, 1 / 2, size=randomguess.size())
                randomguess = randomguess.clamp(0, 1)
            else:
                randomguess = torch.rand_like(yhat)

            randomguess = randomguess.to(device)

            def check_cgr(inferred_value):
                total_branch_num, correct_branch_num = 0, 0
                for i in range(trees_internal_node_features.shape[0]):
                    for j in range(trees_internal_node_features.shape[1]):
                        feature_id = trees_internal_node_features[i, j]
                        if feature_id >= n_attacker:
                            total_branch_num += 1
                            threshold = trees_internal_node_thresholds[i, j]
                            value = inferred_value[feature_id - n_attacker]
                            ground_truth = x[0, feature_id]
                            if (ground_truth <= threshold and value <= threshold) or (
                                    ground_truth > threshold and value > threshold):
                                correct_branch_num += 1

                return total_branch_num, correct_branch_num

            gen_total_num, gen_correct_num = check_cgr(yhat[0, :])
            random_total_num, random_correct_num = check_cgr(randomguess[0, :])

            generator_total_branch_num += gen_total_num
            generator_correct_branch_num += gen_correct_num
            random_total_branch_num += random_total_num
            random_correct_branch_num += random_correct_num

        logging.critical("------------------ SUMMARY ------------------")
        mean_model_cgr = generator_correct_branch_num / generator_total_branch_num
        mean_guess_cgr = random_correct_branch_num / random_total_branch_num
        logging.critical("Mean generator cgr: %s", mean_model_cgr)
        logging.critical("Mean random guess cgr: %s", mean_guess_cgr)
        return mean_model_cgr, mean_guess_cgr

