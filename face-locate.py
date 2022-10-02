# -*- coding: utf-8 -*-

# 检测人脸
import face_recognition
import cv2

# 读取图片并识别人脸
imagepath = 'test/155.jpg'
img = face_recognition.load_image_file(imagepath)
face_locations = face_recognition.face_locations(img)

# 调用opencv函数显示图片
img = cv2.imread(imagepath)
cv2.imshow("Original", img)

# 遍历每个人脸，并标注
for face in face_locations:
    top, right, bottom, left =  face[0:4]

    start = (left, top)
    end = (right, bottom)

    color = (55, 255, 155)
    thickness = 3
    cv2.rectangle(img, start, end, color, thickness)

# 显示识别结果

cv2.imshow("Recogonition", img)

cv2.waitKey(0)
cv2.destroyAllWindows()

