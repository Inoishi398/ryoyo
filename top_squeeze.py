import threading
import time
import os

#Step-1 ___Load sub program___
import video2 as video
import squeezenet as sq

#Step-2 ___sub program excute , video2.py and squeezenet.py
while True:
	os.system('python3 video2.py')
	print("send00")
	time.sleep(1)
	os.system('python3 squeezenet.py')
	print("send01")

	time.sleep(2)

