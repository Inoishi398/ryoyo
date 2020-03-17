#__________________//////////////////////_______________
#__For Video capture function and save as video.png format 
#__Written by Inoishi 2020/Mar/12 from scratch
#__OpenVino version 2019R3
#__Version 1.0 2020/Mar/12
#_________________//////////////////////_______________

#Step-1 ___load plugin for python
import cv2
import sys
import time

camera_id = 0
window_name = 'frame'

#Step-2 ___ Set camera id
cap = cv2.VideoCapture(camera_id)

#Step-3 ___ if no device camera , exit program
if not cap.isOpened():
    sys.exit()

#Step-4 ___ Caputure from camera and save image 'video.png'
#Range 0 - 20 times , meaning exposure adjustment
#print camera data on terminal
for i in range(0,10):
	ret, frame = cap.read()
	cv2.imwrite('video.png', frame)
	print("frame_________________________",frame)

#Step-5 ___ release camera device	
cap.release()

