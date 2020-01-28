# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:27:36 2020

@author: Vsolanki
"""

import emotion_detection
import RecCari
import cv2
from PIL import Image
image_path=r"D:\CV\Face-and-Emotion-Recognition-master\images\Christopher Fabre.jpg"
width = 225
height = 225
m1 = Image.open(image_path)
im5 = m1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
im5.save(image_path , m1.format)

person=RecCari.image_recogination(image_path)
xx=emotion_detection.emotion_recogination(image_path)
tmp_canvas = RecCari.Cartoonizer()
res = tmp_canvas.render(image_path)
cv2.imwrite("Cartoon version1.jpg",res)
cv2.imshow("Cartoon version1.jpg", res)

