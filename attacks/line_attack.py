import cv2
import numpy as np


class LineAttack:
    def __init__(self, color=1.0, thickness=2):
        self.color = color
        self.thickness = thickness

    def generate(self, X, y=None):
        self.c_ = X.shape[1]
        self.h_ = X.shape[2]
        self.w_ = X.shape[3]
        X = np.moveaxis(X, 1, -1)
        outputs = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[0]):
            outputs[i] = self.__draw_line(X[i])
        outputs = np.moveaxis(outputs, -1, 1)
        return outputs

    def __draw_line(self, img):
        color = self.color
        w = self.w_
        h = self.h_
        thickness = self.thickness
        start_x = np.random.randint(low=0, high=w // 2, size=1)
        end_x = np.random.randint(low=w // 2, high=w, size=1)
        y = np.random.randint(low=0, high=h, size=2)
        output = cv2.line(img.copy(), (start_x, y[0]), (end_x, y[1]), color, thickness)
        return output
