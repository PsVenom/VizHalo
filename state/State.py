import cv2
import pyautogui
import tensorflow as tf
from scipy import spatial
import math
from utils.capture import WindowCapture

import numpy as np
pyautogui.FAILSAFE = False
class state(WindowCapture):
    def __init__(self):
        super().__init__(window_name='Halo')

        self.screen_buffer = self.get_screenshot()
        self.health_buffer = self.screen_buffer[43:48, 527:613]
        self.ammo_buffer = self.screen_buffer[38:52, 38:64]
        self.mat1 = np.random.rand(1024)
        self.mat2 = np.random.rand(1024)

    def get_screensample(self):
         im = self.get_screenshot()
         a = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
         mata = a[80:112, 60:92]
         matb = a[360:392, 480:512]

         # try:
         similaritya =  spatial.distance.cosine(mata.flatten().T/255 , self.mat1.T/255)
         similarityb = spatial.distance.cosine(matb.flatten().T/255, self.mat2.T/255)
         cv2.imshow('mata', mata)
         self.mat1 = mata.flatten()
         self.mat2 = matb.flatten()
         return (1-similaritya)* (1-similarityb)
    def get_health(self):
        lower_color_bounds = (101, 83, 97)
        upper_color_bounds = (127, 221, 255)
        mask = cv2.inRange(cv2.cvtColor(self.health_buffer,cv2.COLOR_BGR2HSV), lower_color_bounds, upper_color_bounds)
        return cv2.countNonZero(mask)/(self.health_buffer.shape[0]*self.health_buffer.shape[1])
    def get_ammo(self):
        model_path = 'C:/Users/Dell/PycharmProjects/MasterCheeks/state/ocrhalo.h5'
        occ_model = tf.keras.models.load_model(model_path)
        img = cv2.resize(self.ammo_buffer,
					(28*3, 28),
                    interpolation=cv2.INTER_LANCZOS4)
        im1, im2, im3 = img[:, 0:28], img[:, 28:56], img[:, 56:84]
        return math.fsum([100*np.argmax(occ_model.predict(im1.reshape(-1, 28, 28, 3))),10*np.argmax(occ_model.predict(im2.reshape(-1, 28, 28, 3))), np.argmax(occ_model.predict(im3.reshape(-1, 28, 28, 3)))])


