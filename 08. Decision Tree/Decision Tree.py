# Importing Libraries
import pandas as pd             # Pandas
import numpy as np              # Numpy
import matplotlib.pyplot as plt # Matplotlib.pyplot
import seaborn as sb            # Seaborn
sb.set()                        # Setting Seaborn style


class Node:
    ''' Creates a Node '''
    def __init__(self, col_idx, Decision_Tree_Classifier_object):
        self.col_idx = col_idx
        self.column = Decision_Tree_Classifier_object.X[:, col_idx]
        self.branches = np.unique(self.column)

    def __str__(self):
        return '{} ==> {}'.format(self.col_idx, self.branches)

class Tree:
    ''' Builds a Tree which splits its branches by making decisions '''
    def __init__(self, Node_object):
        ''' Initiates a Tree '''
        self.root_node = Node_object

class Decision_Tree:
    ''' Creates Decisions '''
    pass


class Decision_Tree_Classifier:
    ''' Defines Decision Tree used as a classifier '''
    
    def __init__(self, 
                 visualization = False, 
                 error_metrics = True):
        
        ''' Initalizes the Decision Tree '''
        self.visualization = visualization
        self.error_metrics = error_metrics
        
    def _validate_data(self, X, y):
        
        '''Validates Input Data'''
        
        # Checks Dimensions of Independent columns
        if X.ndim != 2:
            raise Exception('Independent column must have 2 Dimensions.\
                            Got dimension of {}'. format(X.ndim))
        
        # Checks Dimensions of Dependent columns
        elif y.ndim != 1:
            raise Exception('Dependent column must have 1 Dimensions.\
                            Got dimension of {}'. format(y.ndim))
        
        # Checks shape of Independent and Dependent columns
        elif X.shape[0] != y.shape[0]:
            raise Exception('Shape does no match.\
                            Got shape of {} and {}'.
                            format(X.shape, y.shape))
        else:
            pass
        
    def _entropy(self, rows = 'all'):
        ''' Computes Entropy '''
        # Choosing Target based on condition
        _data = self.y.ravel()
        # Choosing rows
        if rows != 'all':
            _data = _data[rows]
        
        # Extracing unique data
        _unique_data = np.unique(_data, return_counts = True)
        
        # Computing Entropy
        _total = len(_data)
        entropy = 0
        for i in range(len(_unique_data[0])):
            # Calculating Probability
            _probability = _unique_data[1][i] / _total
            entropy += (-1) * (_probability) * np.log2(_probability)
        
        return entropy
    
    def _select_node(self, columns):
        ''' Selects Optimum Node '''
        informations = []
        gains = []
        for i in columns:
            uniques = np.unique(X[:, i].ravel(), return_counts = True)
            sum_ = np.sum(uniques[1])
            info = 0
            j = 0
            for unique in uniques[0]:
                rows = np.where(X[:, i] == unique)
                entropy = self._entropy(rows)
                print(entropy, uniques[1][j], sum_)
                info += (entropy * uniques[1][j]) / sum_
                j += 1
            gain = self._info_gain(info)
            informations.append(info)
            gains.append(gain)
        print('INFO : ', informations)
        print('GAIN : ', gains)
            
    
    
    def _info_gain(self, info):
        ''' Returns information gain '''
        return self.primary_entropy - info
    
    def fit(self, X, y):
        ''' Fits Decision Tree '''
        
        # Validate the input data
        self._validate_data(X, y)
        
        # Compute No. of records and No.of Features
        self._m_records, self._n_features = X.shape
        
        # Intializing Independent and Dependent columns
        self.X = X
        self.y = y.reshape(-1, 1)
        
        # Compute Target classes
        self._target_class = np.unique(y.ravel(), return_counts=True)
        
        
        ''' Step 1: Compute the Entropy for the Dataset '''
        self.primary_entropy = self._entropy()
        print(self.primary_entropy)
        
        ''' Step 2: Initializing columns '''
        columns = list(range(self._n_features))
        root_node = self._select_node(columns = columns)
        
        
    
    def predict(self, X):
        ''' Predicts Decision Tree Results '''
        pass