import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import Counter

class AccuracyCounter:
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0
        
    def addOntBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal
    
    def reportAccuracy(self):
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)