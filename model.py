import keras
from keras.models import Model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import load_model

import numpy as np
import tensorflow as tf
import h5py
import keras
from skimage import transform
import skimage, os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
class Cnn_Regression(object):
    def __init__(self, input_shape, njoints):
        self.njoints = int(njoints)
        self.input_shape = input_shape
        self.heatmap_gt = np.empty((int(self.input_shape/8), int(self.input_shape/8), self.njoints), dtype=np.float32)
        
    
    def slice1(self, x, k):
        return x[..., k:k+self.njoints]
    
    def square(self, x):
        return keras.backend.square(x)
    
    def square_root(self, x):
        return keras.backend.sqrt(x)
    
    
    def build_model(self):
        eight_of_input = int(self.input_shape/8)
        input_shape = int(self.input_shape)
        input_tensor = keras.layers.Input(shape=(input_shape,input_shape,3)) ## CHANGE Accordingly
        prev_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
        prev_model = Model(prev_model.input, prev_model.layers[141].output) ## Till res4f
        
        model = Sequential()
        
        b1a = keras.layers.Conv2D(512, 1, strides=(1, 1), activation='relu')(prev_model.output)
        b1a = keras.layers.Conv2D(512, 3, strides=(1, 1), activation='relu', padding='same')(b1a)
        b1a = keras.layers.Conv2D(1024, 1, strides=(1, 1), activation=None)(b1a)
        
        b1b = keras.layers.Conv2D(1024, 1, strides=(1, 1), activation=None)(b1a)
        
        c1 = keras.layers.Add()([b1a, b1b])
        c1 = keras.layers.Activation("relu")(c1)
        
        c1 = keras.layers.Conv2D(256, 1, strides=(1, 1), activation='relu')(c1)
        c1 = keras.layers.Conv2D(128, 3, strides=(1, 1), activation='relu',padding='same')(c1)
        c1 = keras.layers.Conv2D(256, 1, strides=(1, 1), activation='relu')(c1)
       
        b2a = keras.layers.Conv2DTranspose(128, 4, strides=(2, 2), activation='relu', padding="same")(c1)  
        b2a.set_shape([None, eight_of_input, eight_of_input, 128])
        b2b = keras.layers.Conv2DTranspose(self.njoints*3, 4, strides=(2, 2), activation=None, padding= "same")(c1)
        b2b.set_shape([None, eight_of_input, eight_of_input, self.njoints*3])
       
        
        
        b2b_sqr = keras.layers.Lambda(function=self.square, output_shape=(eight_of_input, eight_of_input, self.njoints*3))(b2b)
        
        blx_sqr = keras.layers.Lambda(self.slice1, arguments={'k':int(0)}, output_shape=(eight_of_input,eight_of_input,self.njoints))(b2b_sqr)
        bly_sqr = keras.layers.Lambda(self.slice1, arguments={'k':int(self.njoints)}, output_shape=(eight_of_input,eight_of_input,self.njoints))(b2b_sqr)
        blz_sqr = keras.layers.Lambda(self.slice1, arguments={'k':int(self.njoints*2)}, output_shape=(eight_of_input,eight_of_input,self.njoints))(b2b_sqr)
        print(b2b_sqr)
        print(blx_sqr)
        print(bly_sqr)
        print(blz_sqr)
        
        bl_sqr = keras.layers.Add()([blx_sqr, bly_sqr, blz_sqr])
        
        bl = keras.layers.Lambda(function=self.square_root, output_shape=(eight_of_input, eight_of_input, self.njoints))(bl_sqr)
        print(bl)
        c2 = keras.layers.concatenate([b2a, b2b, bl], axis=-1)
        
        c2 = keras.layers.Conv2D(128, 3, strides=(1, 1), activation='relu', padding="same")(c2)
        c2 = keras.layers.Conv2D(self.njoints*4, 1, strides=(1, 1), activation='relu', padding="same")(c2)
        
        heat_map = keras.layers.Lambda(self.slice1, arguments={'k':int(0)}, output_shape=(eight_of_input,eight_of_input,self.njoints), name="heatmap")(c2)
        locmap_x = keras.layers.Lambda(self.slice1, arguments={'k':int(self.njoints)}, output_shape=(eight_of_input,eight_of_input,self.njoints),name="locmapx")(c2)
        locmap_y = keras.layers.Lambda(self.slice1, arguments={'k':int(self.njoints*2)}, output_shape=(eight_of_input,eight_of_input,self.njoints),name="locmapy")(c2)
        locmap_z = keras.layers.Lambda(self.slice1, arguments={'k':int(self.njoints*3)}, output_shape=(eight_of_input,eight_of_input,self.njoints),name="locmapz")(c2)
        
        print(heat_map)
        print(locmap_x)
        print(locmap_y)
        print(locmap_z)

        self.model = Model(inputs=prev_model.input, outputs=[heat_map, locmap_x, locmap_y, locmap_z])
        #return model
    
    def locloss(self, y_true, y_pred):
        hmap_gt = self.heatmap_gt
        
        locmap_preds = tf.multiply(hmap_gt, y_pred)
        locmap_gt = tf.multiply(hmap_gt, y_true)
        
        loss = keras.losses.mean_squared_error(locmap_gt, locmap_preds)
        return loss
    
    def compile_model(self):
        ada = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=ada, loss={'heatmap': 'mean_squared_error', 'locmapx':self.locloss, 'locmapy':self.locloss, 'locmapz':self.locloss})
        
        return self.model

    


