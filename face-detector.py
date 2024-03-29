
import cv2
import face_recognition as fr
import numpy as np

path,path2,path3,path4 = "ex1","ex2","ex3","ex4"

img = fr.load_image_file(path)  #import images

lst1 = fr.face_landmarks(img) #pick 68 position points in face

lst = fr.face_locations(img,model="CNN")



t,r,b,l = lst[0]  #face positions (top,right,bottom,left)

img2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   #color space rechange

cv2.rectangle(img2,(l,t),(r,b),(255,255,0),6) # sign faces

cv2.imshow("img", img2) # output in new winidow

cv2.waitKey()




