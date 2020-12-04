import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

class LID():
    def __init__(self, model):
        self.model = model
        

    def fit(self, X, X_noisy, X_adv, k=20, batch_size=100):
        # get layer-wise output functions
        pass

    def detect(self, X):
        pass
