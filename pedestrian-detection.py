#__________________//////////////////////_______________
#__For Openvino execute inference engine with device pedestrian-detection
#__Written by Inoishi 2020/Mar/14 from scratch
#__OpenVino version 2019R3
#__Version 1.0 2020/Mar/14
#_________________//////////////////////_______________

#___load plugin for python
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore
from operator import itemgetter
from openvino.inference_engine import IENetwork, IEPlugin

from PIL import Image		#image crop openCV
import time




#___Init xml,bin and image for inference engine
ie = IECore()
__xml="/media/psf/Home/Documents/ubunts/intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml"
__bin="/media/psf/Home/Documents/ubunts/intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.bin"

#__xml="./intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml"
#__bin="./intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.bin"

#__image="perfume.png"
#__image="persons.png"
__image="persons2.png"
#__image="video.png"

__xmax=512 	#if persone over 512 in X , crop_res for security camera and etc...

__plug="CPU"				#option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="GPU"				#option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="MYRIAD"			#option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="HETERO:FPGA,CPU"		#option CPU,GPU,MYRIAD,HETERO:FPGA,CPU

im = Image.open(__image)                                        # Crop from image for openCV


#___Load plugin and cpu extension libraly
#___Load plugin and cpu extension libraly
plugin = IEPlugin(device=__plug)
#if __plug=="CPU":                       #if plug is cpu , add extension for cpu only
#        plugin.add_cpu_extension("/home/ieisw/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so")

#___Set IENetwork with xml , bin
net = IENetwork(model=__xml,weights=__bin)

#___Set exec_net and load to network   
exec_net = plugin.load(network=net)



#___Read image and format for openvino format
img_face = cv2.imread(__image)
img = cv2.resize(img_face, (672,384))		   # for change format face-detection size must 672,384
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))    # HWC > CHW 
img = np.expand_dims(img, axis=0)


#___ execute inference engine with image and get result as "res" start sync mode
res = exec_net.infer(inputs={'data':img}) #infer progress
#___ execute inference engine with image and get result as res end


print("___pedestrian-detection-adas-0002 Start___")
print(res)		#print all result as res , thas is 1000 result 

#___'detection_out' is result from inference engine from res
print("inoishi__________________")
print("The net outputs a blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes")
print("For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]")

detection_out = np.squeeze(res['detection_out'])

#__ Judge if res[2] is over 0.5 , append into values
values=[]
#for index in range(len(res)):
for index in range(len(detection_out)):
	if detection_out[index][2] > 0.5:
		values.append([str(index),detection_out[index][2]])
		print(values[index])

		xmin = int(int(detection_out[index][3] * img_face.shape[1]) * 1.0) #0.9
		ymin = int(int(detection_out[index][4] * img_face.shape[0]) * 1.0) #1.3
		xmax = int(int(detection_out[index][5] * img_face.shape[1]) * 1.0) #1.1
		ymax = int(int(detection_out[index][6] * img_face.shape[0]) * 1.0) #1.3

		print("xmin=",xmin)
		print("ymin=",ymin)
		print("xmax=",xmax)
		print("ymax=",ymax)

                #__ print box for face in picture __
		cv2.rectangle(img_face, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)
		cv2.line(img_face, (__xmax,0),(__xmax,__xmax),color=(255,0,0), thickness=3)

               	#__ Crop face in imag_face and save
		im_crop = im.crop((xmin, ymin, xmax, ymax))
		im_crop.save(str(index)+'crop'+'.png', quality=95)

		#__ For security camera function , if over xmax , save as *crop_res.png
		if xmax > __xmax:
			im_crop.save(str(index)+'crop_res'+'.png', quality=95)


#__show image with face box
cv2.imshow('img_face', img_face)	
cv2.waitKey(5000)
cv2.destroyAllWindows()
cv2.imwrite('result.png', img_face) 


 

#__show inference engine and device(plug)
print()
print("Used plugin device is",__plug)
print("Inference engine is",__xml)
print("___pedestrian-detection-adas-0002 End")


