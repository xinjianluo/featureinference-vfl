import argparse
import configparser
import math
import numpy as np
from random import randint
import logging
import time
import random
import os
from numpy import genfromtxt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
from sklearn.tree import export_text

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

def load_from_csv(data_path, test_perc=0.2, pred_perc=0.2, delimiter=','):
    '''
        assume label on the last feature dimension
        :param test_perc: percentage of data used for validation
        :return:
    '''
    data = genfromtxt(data_path, delimiter=delimiter)
    X, y = data[:, :-1], data[:, -1]
    X_train_test, X_pred, y_train_test, y_pred = train_test_split(X, y, test_size=pred_perc)
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=test_perc)

    # permute the columns such that each trial with different target features
    # the labels do not need permute as the rows are the same
    perm = np.random.permutation(len(X_train.T))
    X_train = X_train.T[perm].T
    X_pred = X_pred.T[perm].T
    X_test = X_test.T[perm].T

    return X_train, y_train, X_test, y_test, X_pred, y_pred

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
    
    p_tree = config['DECISIONTREE']
    parameters['max_depth'] = p_tree.getint('MaxDepth')
    parameters['min_samples_split'] = p_tree.getint('MinSamplesSplit')
    parameters['min_samples_leaf'] = p_tree.getint('MinSamplesLeaf')
    parameters['min_impurity_decrease'] = p_tree.getfloat('MinImpurityDecrease')
    parameters['criterion'] = p_tree['Criterion']
    
    # add time stamp to the name of log file
    logfile = p_default['LogFile']
    index = logfile.rfind('.')
    if index != -1:
        logfile = logfile[:index]  + "_" + "unknown_" + str(parameters['num_target_features']) \
                + "_depth_" + str(parameters['max_depth']) \
                + "_expnum_" + str(parameters['num_exps']) \
                + "_" + time.strftime("%Y%m%d%H%M%S") + logfile[index:]
    else:
        logfile = logfile + "_" + "unknown_" + str(parameters['num_target_features']) \
                + "_depth_" + str(parameters['max_depth']) \
                + "_expnum_" + str(parameters['num_exps']) \
                + "_" + time.strftime("%Y%m%d%H%M%S") + ".log"
        
    parameters['logpath'] = currentDir() + os.sep + "log" + os.sep + logfile
    
    return parameters
    
# The attack method is as follows:
#   1. Split the dataset into three parts for training, testing, prediction (the predict dataset is used for attack)
#   2. Train a decision tree model using the train dataset and test dataset
#   3. Convert the decision tree model into a struct list with a list of nodes, each node includes feature and threshold
#   4. Split the predict dataset into adversary and target
#   5. For each sample in the predict dataset
#       5.1 First compute the ground-truth prediction, i.e., class
#       5.2 Second restrict the candidate prediction paths given the ground-truth class
#       5.3 Third scan each candidate prediction path, and check if it is still possible given adversary's features
#       5.4 Fourth randomly select an available prediction path, and check the target features along with this path
#           record the correct guess number and the total target feature number
#       5.5 Baseline: randomly select a prediction path from all paths, and check the target features along the path
#           record the correct guess number and the total target feature number
#   6. Compare the successful guess rate of the attack and random guess (expectation: attack guess rate is higher)

if __name__=='__main__':  
    manualseed = 50
    random.seed(manualseed)
    np.random.seed(manualseed)
    
    # read parameters from config file
    configfile = 'config.ini'
    parameters = readConfigFile(configfile)
    
    trials = parameters['num_exps']
    
    # init logging
    initlogging(parameters['logpath'])
    # logging.info("This should be only in file") 
    # logging.critical("This shoud be in both file and console")
    
    logging.critical('dataset: %s', parameters['data_path'])
    logging.critical('number of target features: %d', parameters['num_target_features'])
    logging.critical('max tree depth: %d', parameters['max_depth'])
    logging.critical('number of experimental trials: %d', trials)

    # step 1: load and split dataset
    X_train, y_train, X_test, y_test, X_pred, y_pred = load_from_csv(parameters['data_path'], parameters['test_percentage'], parameters['pred_percentage'])
    # print(f'Data shape: \ntrain\t x {X_train.shape},\ty {y_train.shape}\n'
          # f'test\t x {X_test.shape},\ty {y_test.shape}\n'
          # f'pred\t x {X_pred.shape},\ty {y_pred.shape}\n')

    def reload_dateset():
        global X_train, y_train, X_test, y_test, X_pred, y_pred
        X_train, y_train, X_test, y_test, X_pred, y_pred = load_from_csv(parameters['data_path'], parameters['test_percentage'], parameters['pred_percentage'])

    logging.critical('Begin trials')
    path_restriction_accuracy, random_guess_accuracy = AverageMeter(), AverageMeter()
    for perexp in range(trials):
        logging.critical('-------------------------------------')
        logging.critical("Attack trial {} / {}".format(perexp+1, trials))
        # step 2: decision Tree training
        def train():
            assert parameters['criterion'] in ['gini', 'entropy']
            clf = DecisionTreeClassifier(criterion=parameters['criterion'], max_depth=parameters['max_depth'],
                                         min_samples_split=parameters['min_samples_split'], min_samples_leaf=parameters['min_samples_leaf'],
                                         min_impurity_decrease=parameters['min_impurity_decrease'], random_state=randint(0, 1000000))

            reload_dateset()
            clf.fit(X_train, y_train)
            # print('exemplary prediction probabilities: ', clf.predict_proba(X_test[0:3,:]))
            train_perf = clf.score(X_train, y_train)
            test_perf = clf.score(X_test, y_test)
            r = export_text(clf, show_weights=True)
            # print('tree structure:\n', r)
            # print('tree node count: ', clf.tree_.node_count)
            # print(f'train performance: {train_perf}, test performance: {test_perf}')

            logging.info(f'Decision Tree\t {trials}-Avg train_perf:\t{train_perf:4f}, test_perf:\t{test_perf:4f}')
            return clf

        # step 3: convert the trained decision tree model to list
        def convert(clf):
            tree_ = clf.tree_
            node_num = int(math.pow(2, tree_.max_depth + 1) - 1)

            # initialize four arrays to store the necessary information
            node_types = [-1] * node_num
            leaf_labels = [-1] * node_num
            internal_node_features = [-1] * node_num
            internal_node_thresholds = [-1] * node_num

            # recursive traverse tree_ and store info in the four arrays
            def tree_recurse(tree_node, full_node_id, depth):
                value = tree_.value[tree_node][0]
                class_name = np.argmax(value)
                if depth <= tree_.max_depth + 1:
                    if tree_.feature[tree_node] != _tree.TREE_UNDEFINED:
                        node_types[full_node_id] = 0
                        threshold = tree_.threshold[tree_node]
                        feature = tree_.feature[tree_node]
                        internal_node_features[full_node_id] = feature
                        internal_node_thresholds[full_node_id] = threshold
                        tree_recurse(tree_.children_left[tree_node], 2 * full_node_id + 1, depth + 1)
                        tree_recurse(tree_.children_right[tree_node], 2 * full_node_id + 2, depth + 1)
                    else:
                        node_types[full_node_id] = 1
                        leaf_labels[full_node_id] = class_name

            tree_recurse(0, 0, 1)
            #print(node_types)
            #print(leaf_labels)
            #print(internal_node_features)
            #print(internal_node_thresholds)
            return node_types, leaf_labels, internal_node_features, internal_node_thresholds

        clf = train()
        full_node_types, full_leaf_labels, full_internal_features, full_internal_thresholds = convert(clf)

        # step 4: split the prediction dataset into adversary and target
        total_feature_num = X_pred.shape[1]
        known_feature_num = total_feature_num - parameters['num_target_features']
        target_feature_idx = np.arange(known_feature_num, total_feature_num)
        logging.info('target feature indexes: %s', str(target_feature_idx))

        # step 5: for each sample in X_pred, do path restriction attack
        full_node_num = int(math.pow(2, clf.tree_.max_depth + 1)) - 1
        total_unknown_comp_num = 0
        correct_guess_comp_num = 0
        baseline_total_unknown_comp_num = 0
        baseline_correct_guess_comp_num = 0
        #debug_X_pred = X_pred[0:3, :]
        for sample in X_pred:
            # 5.1 First compute the ground-truth prediction, i.e., class
            # 5.2 Second restrict the candidate prediction paths given the ground-truth class
            # 5.3 Third scan each candidate prediction path, and check if it is still possible given adversary's features
            # 5.4 Fourth randomly select an available prediction path, and check the target features along with this path
                # record the correct guess number and the total target feature number
            # 5.5 Baseline: randomly select a prediction path from all paths, and check the target features along the path
                # record the correct guess number and the total target feature number
            sample_2d = np.reshape(sample, (-1, total_feature_num))
            # print('sample values: ', sample_2d)
            label = clf.predict(sample_2d)

            # the first_filter is based on the predicted label
            first_filter = (full_leaf_labels == label[0]) + 0
            # print(first_filter)

            # scan the tree and compute the binary array, if a path is feasible according to
            # the adversary's features, then the corresponding element is 1, otherwise is 0
            second_filter = np.ones_like(full_leaf_labels)
            def update_recurse(node_id):
                if node_id >= full_node_num + 1:
                    return
                else:
                    # retrieve feature and threshold on the current node
                    cur_feature = full_internal_features[node_id]
                    cur_threshold = full_internal_thresholds[node_id]
                    if cur_feature == -1:
                        return
                    if cur_feature in target_feature_idx:
                        second_filter[2 * node_id + 1] = second_filter[node_id] * 1
                        second_filter[2 * node_id + 2] = second_filter[node_id] * 1
                    else:
                        if sample[cur_feature] <= cur_threshold:
                            # left children is candidate
                            second_filter[2 * node_id + 1] = second_filter[node_id] * 1
                            second_filter[2 * node_id + 2] = second_filter[node_id] * 0
                        else:
                            # right children is candidate
                            second_filter[2 * node_id + 1] = second_filter[node_id] * 0
                            second_filter[2 * node_id + 2] = second_filter[node_id] * 1

                    update_recurse(2 * node_id + 1)
                    update_recurse(2 * node_id + 2)

            update_recurse(0)
            # print(second_filter)

            candidate_paths = np.multiply(first_filter, second_filter)
            count_candidate_paths = sum(x == 1 for x in candidate_paths)
            if count_candidate_paths == 0:
                continue
            # randomly choose a candidate path and check the correct guess rate
            candidate_paths_probs = candidate_paths / count_candidate_paths
            choose_index_pra = np.random.choice(full_node_num, 1, p=candidate_paths_probs)
            # print(candidate_paths)
            # print('candidate path num: ', count_candidate_paths)
            # print('candidate path probabilities: ', candidate_paths_prob)

            cond = np.zeros(full_node_num)
            candidate_paths_rg = (full_leaf_labels >= cond) + 0
            count_candidate_paths_rg = sum(x == 1 for x in candidate_paths_rg)
            if count_candidate_paths_rg == 0:
                continue
            candidate_paths_probs_rg = candidate_paths_rg / count_candidate_paths_rg
            choose_index_rg = np.random.choice(full_node_num, 1, p=candidate_paths_probs_rg)
            # print('selected path index in path restriction attack: ', choose_index_pra[0])
            # print('selected path index in random guess: ', choose_index_rg[0])

            # given prediction path index, back-trace the tree and check
            check_node_id = choose_index_pra[0]
            while check_node_id >= 0:
                # print('check node id: ', check_node_id)
                parent_node_id = int((check_node_id - 1)/2)
                # print('parent node id: ', parent_node_id)
                feature_id = full_internal_features[parent_node_id]
                # print('feature id: ', feature_id)
                if feature_id in target_feature_idx:
                    feature_threshold = full_internal_thresholds[parent_node_id]
                    # print('feature threshold: ', feature_threshold)
                    ground_truth_feature_value = sample[feature_id]
                    # print('ground truth feature value: ', ground_truth_feature_value)
                    total_unknown_comp_num += 1
                    if (ground_truth_feature_value <= feature_threshold) and (check_node_id == 2 * parent_node_id + 1):
                        # correct guess on the left partition
                        correct_guess_comp_num += 1
                    elif (ground_truth_feature_value > feature_threshold) and (check_node_id == 2 * parent_node_id + 2):
                        # correct guess on the right partition
                        correct_guess_comp_num += 1
                    else:
                        # guess wrong
                        correct_guess_comp_num += 0

                if parent_node_id == 0:
                    break
                else:
                    check_node_id = parent_node_id

            # given prediction path index
            check_node_id_rg = choose_index_rg[0]
            while check_node_id_rg >= 0:
                # print('check node id rg: ', check_node_id_rg)
                parent_node_id_rg = int((check_node_id_rg - 1)/2)
                # print('parent node id rg: ', parent_node_id)
                feature_id_rg = full_internal_features[parent_node_id_rg]
                # print('feature id rg: ', feature_id_rg)
                if feature_id_rg in target_feature_idx:
                    feature_threshold_rg = full_internal_thresholds[parent_node_id_rg]
                    # print('feature threshold rg: ', feature_threshold_rg)
                    ground_truth_feature_value_rg = sample[feature_id_rg]
                    # print('ground truth feature value rg: ', ground_truth_feature_value_rg)
                    baseline_total_unknown_comp_num += 1
                    if (ground_truth_feature_value_rg <= feature_threshold_rg) and (check_node_id_rg == 2 * parent_node_id_rg + 1):
                        # correct guess on the left partition
                        baseline_correct_guess_comp_num += 1
                    elif (ground_truth_feature_value_rg > feature_threshold_rg) and (check_node_id_rg == 2 * parent_node_id_rg + 2):
                        # correct guess on the right partition
                        baseline_correct_guess_comp_num += 1
                    else:
                        # guess wrong
                        baseline_correct_guess_comp_num += 0

                if parent_node_id_rg == 0:
                    break
                else:
                    check_node_id_rg = parent_node_id_rg

        path_restriction_rate, random_guess_rate = None, None
        logging.critical('total correct guess num: %d', correct_guess_comp_num)
        logging.critical('total unknown compare num: %d', total_unknown_comp_num)
        if total_unknown_comp_num != 0:
            path_restriction_rate = correct_guess_comp_num/total_unknown_comp_num
            logging.critical('total correct guess rate: %f', path_restriction_rate)

        logging.critical('baseline total correct guess num: %d', baseline_correct_guess_comp_num)
        logging.critical('baseline total unknown compare num: %d', baseline_total_unknown_comp_num)
        if baseline_total_unknown_comp_num != 0:
            random_guess_rate = baseline_correct_guess_comp_num/baseline_total_unknown_comp_num
            logging.critical('baseline total correct guess rate: %f', random_guess_rate)

        if path_restriction_rate is not None and random_guess_rate is not None:
            path_restriction_accuracy.update(path_restriction_rate)
            random_guess_accuracy.update(random_guess_rate)

    logging.critical('-------------------------------------')
    logging.critical(f'Path restriction attack\t {trials}-Avg accuracy:\t{path_restriction_accuracy.avg:4f}')
    logging.critical(f'Random guess attack\t {trials}-Avg accuracy:\t{random_guess_accuracy.avg:4f}')
    print("See {} for more details.".format(parameters['logpath']))