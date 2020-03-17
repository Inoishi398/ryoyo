import threading
import threading
import time

import os
import time

#___Load sub program___
import video2 as video
import face_detect_crop as face
import age_gender_async as age

while True:
	os.system('python3 video2.py')
	print("send00")
	time.sleep(1)
	os.system('python3 face_detect_crop.py')
	os.system('python3 age_gender_async.py')
	print("send01")

	top_age=age.age
	top_female=age.prob[0]
	top_male=age.prob[1]
	print("Top_age_____",top_age)
	print("Top_female__",top_female)
	print("Top_male____",top_male)
	#time.sleep(1.5)

