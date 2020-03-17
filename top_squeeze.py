import threading
import threading
import time

import os
import time

#___Load sub program___
import video2 as video
import squeezenet as sq

while True:
	os.system('python3 video2.py')
	print("send00")
	time.sleep(1)
	os.system('python3 squeezenet.py')
	print("send01")

	time.sleep(2)

