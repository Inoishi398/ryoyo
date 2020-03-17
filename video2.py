#__________________//////////////////////_______________
#__For Video capture function and save as video.png format 
#__Written by Inoishi 2020/Mar/12 from scratch
#__OpenVino version 2019R3
#__Version 1.0 2020/Mar/12
#_________________//////////////////////_______________

#___load plugin for python
import cv2
import sys
import time

camera_id = 0
delay = 1
window_name = 'frame'

cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    sys.exit()

#while True:
for i in range(0,20):
	ret, frame = cap.read()
	#cv2.imshow(window_name, frame)
	cv2.imwrite('video.png', frame)
	#cv2.imshow('video.png', frame)
	print("frame_________________________",frame)
	
print("wait__________________________",frame)
#cv2.imshow('video.png', frame)
#time.sleep(2)
#cv2.destroyWindow(window_name)
#cv2.destroyWindow('video.png')
cap.release()

