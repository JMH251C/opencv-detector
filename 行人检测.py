import cv2



img = cv2.imread("/home/jmh/4.jpg")

bodyCascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

bodys = bodyCascade.detectMultiScale(img,1.15)

for (x,y,w,h) in bodys:
    cv2.rectangle(img,(x,y),(x + w,y + h),(0,0,255),5)

    cv2.imshow("img",img)
    cv2.waitKey()