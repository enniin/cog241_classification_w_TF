## Parse csv-data in form
## (float, float, ..., float, str)
## into numpy arrays for use with TensorFlow.

import numpy as np
import random

class ParseCsv(object):
    
    ## @param  src -- source file path
    ## @param  labels -- labels as a string list
    ## @param  y_last -- if False, y is first column, default is last (True)
    def __init__(self, src, labels,  y_last=True):
        rawdata = list()
        self.labels = labels
        # Try to open file, else exit
        try:
            fh = open(src, 'r')
        except:
            print("No such file!")
            return None
        # File ok, let's parse:
        for line in fh:
            rawdata.append(line.rstrip().split(','))
        ## -------------------------------------------- ##
        ## One numpy array for all data is generated here.
        ## Actual batches for training and testing are
        ## created by calling getBatch().
        ## -------------------------------------------- ##
        self.data = self._parseNpArray(rawdata, y_last)
        self.examples = len(self.data)
        
    
    ## @param  n_inst -- num. of instances, if all, use None
    ## @param  shuffle -- with shuffling or no, default is yes
    ## @param  hotone -- targets (ys) hot-one or not, default is yes
    ## @param  normalize -- normalize parameters or not, default is no
    ## @param  tratio -- ratio of training set size to test set, default is 10
    def getBatch(self, n_inst, shuffle=True, hotone=True, normalize=False, tratio=10):
        if n_inst is None:
            n_inst = self.examples
        data_dup = np.copy(self.data)
        yi = -1
        nx = len(data_dup[0])+yi
        # If shuffle, shuffle:
        if shuffle:
            np.random.shuffle(data_dup)
        # Split into x & y
        xs = np.copy(data_dup[:n_inst,:nx])
        if hotone:
            ys = np.zeros((n_inst, len(self.labels)))
            for i in range(n_inst):
                lab = int(data_dup[i,yi])
                ys[i, lab] = 1.0
        else:
            ys = np.copy(data_dup[:n_inst,yi])
        # Normalize:
        if normalize:
            xs = self._getNormalized(xs)
        # Split into train & test:
        n_test = int(n_inst/tratio)
        x_test = xs[:n_test]
        y_test = ys[:n_test]
        x_train = xs[n_test:]
        y_train = ys[n_test:]
        return (x_train, y_train, x_test, y_test)
        
    
    ## -------- Helper functions - do not call! -------- ##
    def _parseNpArray(self, rawdata, y_last):
        nparam = len(rawdata[0])
        ndata = len(rawdata)
        data = np.empty((ndata, nparam))
        if y_last:
            yi = -1
        else:
            yi = 0
        i = 0
        for row in rawdata:
            # parameters:
            j = 1+yi
            for s in row[j:nparam+yi]:
                try:
                    data[i,j-1-yi] = float(s)
                except:
                    print('NAN')
                j = j+1
            # targets:
            k = 0
            for lab in self.labels:
                if row[yi] == lab:
                    data[i,-1] = k
                k = k+1
            i = i+1
        return data
    
    def _getNormalized(self, a):
        sums = a.sum(axis=0)
        n = len(a)
        means = sums/n
        acentr = a - means
        l2 = np.atleast_1d(np.linalg.norm(acentr, 2, 0))
        l2[l2==0] = 1
        return acentr / np.expand_dims(l2, 0)
    

    