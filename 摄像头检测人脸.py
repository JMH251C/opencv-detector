import cv2
import face_recognition


capture = cv2.VideoCapture(0)
retval,img = capture.read()

#img = cv2.imread("/home/jmh/3.jpg")

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while capture.isOpened():

    r,i = capture.read()
    faces = faceCascade.detectMultiScale(i,1.5,5)
    if len(faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(i, (x, y), (x + w, y + h), (0, 70, 255), 5)

    cv2.imshow("img", i)
    cv2.waitKey(1)













