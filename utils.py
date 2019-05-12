#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:18:21 2019

@author: fiona06
"""
import os
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import mnist, fashion_mnist
import cv2

slim = tf.contrib.slim

def check_folder(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return dir
    
def dataGenerator(images, labels, data_name, batch_size, imgH=None, imgW=None):
    total = labels.shape[0]   
    num_batch = total // batch_size
    idx = 0
    
    index = np.asarray(range(total))
    np.random.shuffle(index)
    images = images[index]
    labels = labels[index]
    
    if data_name == 'mnist' or data_name == 'fashion-mnist':
        while True:
            #do the shuffling
            if idx == num_batch:
                np.random.shuffle(index)
                images = images[index]
                labels = labels[index]
                idx = 0
            
            yield images[idx*batch_size:(idx+1)*batch_size], labels[idx*batch_size:(idx+1)*batch_size]
            idx += 1
    elif data_name == 'anime':
        while True:
            #do the shuffling
            if idx == num_batch:
                np.random.shuffle(index)
                images = images[index]
                labels = labels[index]
                idx = 0
            
            yield read_images(images[idx*batch_size:(idx+1)*batch_size], imgH, imgW), labels[idx*batch_size:(idx+1)*batch_size]
            idx += 1
        
def read_images(fileNames, imgH, imgW):
    X = []
    for fn in fileNames:
        image = cv2.cvtColor(
                cv2.resize(
                        cv2.imread(fn, cv2.IMREAD_COLOR),
                        (imgW, imgH)),
                cv2.COLOR_BGR2RGB)
        X.append(image)
    return np.asarray(X, dtype = np.float32) / 127.5 - 1
        
def load_mnist(data_name):
    if data_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif data_name == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise NotImplementedError
   
    x = np.concatenate((x_train, x_test), axis = 0).astype(np.float32)
    y = np.concatenate((y_train, y_test), axis = 0)
    
    #one hot encoding
    y = keras.utils.to_categorical(y, 10).astype(np.float32)
    #scale to [-1, 1]
    x = (x / 127.5) - 1
    x = np.expand_dims(x, axis = 3)
    return x, y

def load_anime(dataset_name):
    data_dir = os.path.join('../dataset', dataset_name)
    tag_csv_filename = os.path.join(data_dir, 'tags.csv')
    tag_dict = ['orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair',
                'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair', 
                'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes',
                'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
    tag_csv = open(tag_csv_filename, 'r').readlines()

    label_list = []
    filename_list = []
    for line in tag_csv:
        fid, tags = line.split(',')
        label = np.zeros(len(tag_dict))
        
        for i in range(len(tag_dict)):
            if tag_dict[i] in tags:
                label[i] = 1
        
        label_list.append(label)
        image_path = os.path.join(data_dir,'images',fid+'.jpg')
        filename_list.append(image_path)
    
    return np.array(filename_list), np.asarray(label_list, dtype=np.float32)

#combine all figures in one figure    
def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    
    images = inverse_transform(images)
    img_h, img_w, num_channel = images.shape[1:]
    n_figs = int(np.ceil(np.sqrt(images.shape[0])))
    
    if num_channel == 1:
        images = np.squeeze(images, axis=(3,))
        m = np.ones((n_figs*img_h + n_figs + 1, n_figs*img_w + n_figs + 1)) * 0.5 #here add grid
    else:
        m = np.ones((n_figs*img_h + n_figs + 1, n_figs*img_w + n_figs + 1, 3)) * 0.5 #here add grid
    
    row_start = 1
    for x in range(n_figs):
        col_start = 1
        row_end = row_start + img_h
        for y in range(n_figs):
            index = x*n_figs + y
            col_end = col_start + img_w
            if index < images.shape[0]:
                m[row_start:row_end, col_start:col_end] = images[index]
            col_start = col_end + 1
        row_start = row_end + 1
    
    m = (m * 255.).astype(np.uint8)
    return m


def inverse_transform(images):
    return (images + 1.) / 2

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
            
    
    
    