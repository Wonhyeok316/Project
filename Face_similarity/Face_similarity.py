# https://saiashish90.medium.com/facial-similarity-using-opencv-and-dlib-dc03f745cf10

import dlib
import numpy as np
import cv2
import matplotlib.pyplot as plt

def getFace(img):
    face_detector = dlib.get_frontal_face_detector()
    return face_detector(img, 1)[0]

def encodeFace(image):
    face_location = getFace(image)
    pose_predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
    face_landmarks = pose_predictor(image, face_location)
    face_encoder = dlib.face_recognition_model_v1('model/dlib_face_recognition_resnet_model_v1.dat')
    face = dlib.get_face_chip(image, face_landmarks)
    encodings = np.array(face_encoder.compute_face_descriptor(face))
    return encodings

def getSimilarity(image1, image2):
    face1_embeddings = encodeFace(image1)
    face2_embeddings = encodeFace(image2)
    return np.linalg.norm(face1_embeddings-face2_embeddings)

img1 = cv2.imread('data/Eunbin.PNG')
img2 = cv2.imread('data/IU.PNG')

distance = getSimilarity(img1,img2)
print(distance)

if distance < .6:
    print("Faces are of the same person.")
else:
    print("Faces are of different people.")
  
img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# 이미지를 화면에 표시합니다.
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img1_rgb)
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2_rgb)
plt.title('Image 2')
plt.axis('off')

plt.show()
  