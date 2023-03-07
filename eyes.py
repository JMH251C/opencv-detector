import cv2


img = cv2.imread("/home/jmh/00seg3.jpg")

eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

eyes = eyesCascade.detectMultiScale(img,1.15)

for (x,y,w,h) in eyes:
    cv2.rectangle(img,(x,y),(x + w,y + h),(0,70,255),5)


cv2.imshow("img",img)
cv2.waitKey()



