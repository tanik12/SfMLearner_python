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
cap = cv2.VideoCapture(0)
#cap2 = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 60)           # カメラFPSを60FPSに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 440) # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180) # カメラ画像の縦幅を720に設定

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    # スクリーンショットを撮りたい関係で1/4サイズに縮小
    frame = cv2.resize(frame, ( (int(416), int(128))) )

    pred = sfm.inference(frame[None,:,:,:], sess, mode='depth')

    # 加工なし画像を表示する
    cv2.imshow('Raw Frame', frame)
    # depth画像表示
    #cv2.imshow('Raw Frame222222222', normalize_depth_for_display(pred['depth'][0,:,:,0]))
    #img2 = cv2.imread(normalize_depth_for_display(pred['depth'][0,:,:,0]).astype(unit8))
    cv2.imshow('Raw Frame33333333333333', normalize_depth_for_display(pred['depth'][0,:,:,0]))
    
    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
