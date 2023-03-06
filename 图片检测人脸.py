import cv2

img = cv2.imread("/home/jmh/3.jpg") #img path

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = faceCascade.detectMultiScale(img,1.15)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x + w,y + h),(0,0,255),5)

    cv2.imshow("img",img)
    cv2.waitKey()
