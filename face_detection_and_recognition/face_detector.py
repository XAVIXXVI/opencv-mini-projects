import cv2 as cv

img = cv.imread("photos/group 1.jpg")
cv.imshow("Lady image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image", gray)

haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                           minNeighbors=1)
print("Number of faces found: ", len(faces_rect))

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow("Detected faces in the Image", img)

cv.waitKey(0)
