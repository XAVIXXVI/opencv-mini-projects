import cv2 as cv
# import numpy as np
import os

haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

people = []
for i in os.listdir(r"D:\ML_projects\workspace\opencv-mini-projects\face_detection_and_recognition\faces\train"):
    people.append(i)

# features = np.load("features.npy", allow_pickle = True)
# labels = np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread(r"D:\ML_projects\workspace\opencv-mini-projects\face_detection_and_recognition\faces\val\elton_john\1.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Unidentified image", gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f"Label:  {people[label]} with confidence of {confidence}")
    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow("Detected face", img)

cv.waitKey(0)
