#!/usr/bin/python
# -*- coding: utf-8 -*-
import cv2
#from __future__ import division
import os
import numpy as np
import PIL.Image as pil
import tensorflow as tf
from SfMLearner import SfMLearner
from utils import normalize_depth_for_display
tf.reset_default_graph()
import matplotlib.pyplot as plt

img_height=128
img_width=416
#img_height=960
#img_width=1280
ckpt_file = 'models/model-190532'
sfm = SfMLearner()
sfm.setup_inference(img_height,
                    img_width,
                    mode='depth')

saver = tf.train.Saver([var for var in tf.model_variables()]) 

with tf.Session() as sess:
    saver.restore(sess, ckpt_file)

    # VideoCaptureのインスタンスを作成する。
    # 引数でカメラを選べれる。
    ###cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('test.mp4')   
 
    cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # カメラ画像の縦幅を720に設定
    
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 440) # カメラ画像の横幅を1280に設定
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180) # カメラ画像の縦幅を720に設定
    
    while True:
        # VideoCaptureから1フレーム読み込む
        ret, frame = cap.read()
    
        frame = cv2.resize(frame, ( (int(416), int(128))) )
        pred = sfm.inference(frame[None,:,:,:], sess, mode='depth')
    
        # 加工なし画像を表示する
        frame = cv2.resize(frame, ( (int(1248), int(384))) )
        cv2.imshow('Raw Frame', frame)

        # depth画像表示
        #cv2.imshow('Raw Frame222222222', normalize_depth_for_display(pred['depth'][0,:,:,0]))
        #img2 = cv2.imread(normalize_depth_for_display(pred['depth'][0,:,:,0]).astype(unit8))

        frame2 = cv2.resize(normalize_depth_for_display(pred['depth'][0,:,:,0]), ( (int(1248), int(384))) )
        cv2.imshow('SfMLearner', frame2)
        #cv2.imshow('Raw Frame33333333333333', normalize_depth_for_display(pred['depth'][0,:,:,0]).resize(int(480), int(640)))
        
        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27:
            break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
