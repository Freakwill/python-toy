#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将指定文件夹(默认为同级路径的images/)中的图片中的人脸识别出来

人脸图片保存在同级路径的 faces/ 文件夹中
"""


import pathlib

import face_recognition
import cv2

FACE_PATH = pathlib.Path('faces')

for imagepath in pathlib.Path('images').iterdir():
    if imagepath.suffix=='.jpg':
        img = face_recognition.load_image_file(imagepath)
    else:
        continue
    face_locations = face_recognition.face_locations(img)
    img = cv2.imread(str(imagepath))

    # 遍历每个人脸，并标注
    if not FACE_PATH.exists():
        FACE_PATH.mkdir(parents=True)
    for k, face in enumerate(face_locations, 1):
        top, right, bottom, left =  face[0:4]

        start = (left, top)
        end = (right, bottom)

        color = (55, 255, 155)
        thickness = 3
        # cv2.rectangle(img, start, end, color, thickness) # label it
        face = img[top:bottom, left:right]
        name = f'{FACE_PATH / imagepath.stem}-{k}.jpg'

        cv2.imwrite(name, face)

