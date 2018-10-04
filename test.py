#import keras
#from keras.datasets import mnist
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
#import tensorflow as tf

from layers import *
from tensorflow.python.client import device_lib
local_devices = device_lib.list_local_devices()
print ([x.name for x in local_devices if x.device_type == 'XLA_GPU'])

import os
import glob
import sys
import numpy as np
import random
import cv2
#import matplotlib.pyplot as plt
import pandas
import pydicom
import dicom
from pydicom.data import get_testdata_files
from imgaug import augmenters as iaa
import pandas as pd
ROOT_DIR = os.getcwd()
train_dicom_dir = os.path.join(ROOT_DIR, 'stage_1_train_images')
 
batch_size =16
num_classes = 1
epochs = 100
DIM = 1024
# input image dimensions
img_rows, img_cols = 1024,1024
 
# split the data here into training and testing data.
def get_dicom_fps(dicom_dir):
    dicom_imgs = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_imgs))
def parse_dataset(dicom_dir, annotations):
    #This function returns the file pointer to the images, along with an array of the corresponding labels(patient ID)
    #creates a dictionary, image_fps is a dictionary
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp:[] for fp in image_fps}
    for index, rows in annotations.iterrows():
        fp = os.path.join(dicom_dir, rows['patientId']+'.dcm')
        image_annotations[fp].append(rows)
    return image_fps, image_annotations
annotations = pd.read_csv(os.path.join(ROOT_DIR, 'stage_1_train_labels.csv'))
image_fps, image_annotations = parse_dataset(train_dicom_dir, annotations=annotations)
print(type(image_fps))
#if K.image_data_format() == 'channels_first':
#    print("HI")
#else:
#    print("Bye")
input_shape = (img_rows, img_cols, 1)
# convert class vectors to binary class matrices
'''
model = Sequential()
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, kernel_size=(3, 3),activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu',input_shape=input_shape))
model.add(Dense(num_classes, activation='linear'))
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
s = 0.0
#tot_epoch = 0.0
for reps in range(10):
    for iterepo in range(23*10):
        #s = 0.0
        for i in range(epochs):
            a = []
            b = []
 
            for j in range(batch_size):
                k =image_fps[iterepo*epochs+i+j]
                df=pydicom.read_file(k)
                image = df.pixel_array
                image = np.reshape(image,(1024,1024,1))
                a.append(image)
                b.append(image_annotations[k][0][5])
            x_train = np.asarray(a)
            y_train = np.asarray(b)
 
            history = model.fit(x_train, y_train,batch_size=batch_size,epochs=1,verbose=0)
            #lo =history.history['loss'][0]
            acc = history.epoch
            print(history.history)
            #print (lo)
            #if lo >1:
            #    s +=0
            #else:
            #    s +=1
        #tot_epoch = tot_epoch+epochs
        #print(s/tot_epoch)
        #print(acc)
    print("hi")
'''
#train test
'''
a = []
b = []
for j in range(batch_size):
    k =image_fps[23000+j]
    df=pydicom.read_file(k)
    image = df.pixel_array
    if len(image.shape) != 3 or image.shape[2] != 3:
        image = np.stack((image,) * 3, -1)
    a.append(image)
    b.append(image_annotations[k][0][5])
x_test = np.asarray(a)
y_test = np.asarray(b)
 
print(x_test)
print(y_test)
score = model.evaluate(x_test, y_test, verbose=1)
 
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
with tf.device('/device:XLA_GPU:0'):
    inp = tf.placeholder(tf.float32, shape=[batch_size, img_rows, img_cols, 1])
    gt = tf.placeholder(tf.float32, shape=[batch_size, img_rows, img_cols, 1])
    '''
    pool1 = max_pool_2x2(inp)
    pool1 = max_pool_2x2(pool1)
    conv1 = conv_layer(pool1, shape=[3,3,1,16])
    conv1 = conv_layer(conv1, shape=[3,3,16,16])
    pool2 = max_pool_2x2(conv1)
    conv2 = conv_layer(pool2, shape=[3,3,16,64])
    conv2 = conv_layer(conv2, shape=[3,3,64,64])
    conv2 = conv_layer(conv2, shape=[3,3,64,64])
    pool3 = max_pool_2x2(conv2)
    pool3 = max_pool_2x2(pool3)
    conv3 = conv_layer(pool3, shape=[3,3,64,96])
    conv3 = conv_layer(conv3, shape=[3,3,96,32])
    pool4 = max_pool_2x2(conv3)
    flat = tf.reshape(pool4, [-1,16*16*32])
    full1 = tf.nn.relu(full_layer(flat, 128))
    full1 = tf.nn.sigmoid(full_layer(full1, batch_size))
    '''

    #encoder
    net = conv_layer(inp,shape =[5,5,1,16], name = 'c1') #bsx1024x1024x16
    net = conv_layer(net,shape =[5,5,16,32], name = 'c2') #bsx1024x1024x32
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2) #bsx512x512x32
    net = conv_layer(net,shape =[5,5,32,64], name = 'c3') #bsx512x512x64
    net = conv_layer(net,shape =[5,5,64,64], name = 'c4') #bsx512x512x64
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2) #bsx256x256x64
    net = conv_layer(net,shape =[5,5,64,128], name = 'c5')                              #bsx256x256x64
    net = conv_layer(net,shape =[5,5,128,128], name = 'c6')                             #bsx256x256x128
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2) #bsx128x128x128
    net = conv_layer(net,shape =[5,5,128,128], name = 'c8')                             #bsx128x128x128
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[8, 8], strides=8) #bsx16x16x128
    print(net)
    #decoder
    net = tf.reshape(net,shape = [-1,16*16*128])
    net = tf.layers.dense(net,units = 2**12)
    net = tf.reshape(net,shape = [-1,64,64,1])
    #net = tf.nn.conv2d_transpose(net,filter = [3,3,8,1],output_shape = [1,256,256,8], strides = [1,4,4,1])
    net = tf.layers.conv2d_transpose(net,8,5, strides = 4, padding = 'SAME')
    #net = tf.nn.conv2d_transpose(net,filter = [3,3,16,8],output_shape = [1,512,512,16], strides = [1,2,2,1])
    net = tf.layers.conv2d_transpose(net,16,5, strides = 2, padding = 'SAME')
    print(net)
    #ground_truth = tf.nn.conv2d_transpose(net,filter = [3,3,batch_size,16],output_shape = [1,1024,1024,batch_size], strides = [1,2,2,1]) 
    ground_truth = tf.layers.conv2d_transpose(net,1, 5, strides = 2, padding = 'SAME') 
    
    print(ground_truth)

#ground_truth = tf.reshape(ground_truth, shape = [batch_size, 1024, 1024, 1])
'''
'''

#loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=full1, labels =gt)
loss = tf.reduce_sum(-tf.multiply(gt,tf.log(tf.sigmoid(ground_truth)))-tf.multiply(1-gt,tf.log(1-tf.sigmoid(ground_truth))))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

accuracy = (gt*ground_truth >= 0.5)

#print(tf.size(accuracy))
#accuracy = tf.reduce_mean(accuracy)
#_sum = 0
#for i in range(tf.size(accuracy)):
#    if accuracy[i]==1:
#        _sum = _sum+1

#acc = (_sum*1.0)/tf.size(accuracy)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print('hi')
    sess.run(tf.global_variables_initializer())
    a = []
    b = []
    x = y = z = 0
    for reps in range(10):
        for iter_rep in range(230):
            print('iter: ', iter_rep)
            for i in range(epochs):
                #print('hi'+str(i))
                a = []
                b = []
                for j in range(batch_size):
                    k =image_fps[iter_rep*epochs+i+j]
                    df=pydicom.read_file(k)
                    image = df.pixel_array
                    image = np.reshape(image,(1024,1024,1))
                    a.append(image)
                    path = "./ground/"
                    n = cv2.imread(path + image_annotations[k][0][0]+ ".png",0)
                    n = np.reshape(n, (1024,1024,1))
                    b.append(n)
                x_train = np.asarray(a)
                y_train = np.asarray(b)
                print('running session')
                z,x,y = sess.run([optimizer, loss, accuracy], feed_dict={inp:x_train, gt:y_train})
                print ('loss: '+str(x))
            print (x)


