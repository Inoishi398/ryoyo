#__________________//////////////////////_______________
#__For Openvino execute inference engine with device face-detection 
#__Written by Inoishi 2020/Mar/5 from scratch
#__OpenVino version 2019R3
#__Version 1.0 2020/Mar/11
#__Version 1.1 2020/Mar/12
#_________________//////////////////////_______________

#Step-1 ___load plugin for python
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




#Step-2 ___Init xml,bin and image for inference engine
ie = IECore()
__xml="/media/psf/Home/Documents/ubunts/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml"
__bin="/media/psf/Home/Documents/ubunts/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.bin"
#__image="kanna.png"
#__image="0crop.png"
__image="perfume.png"
#__image="brad.png"
#__image="video.png"

__plug="CPU"				#option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="GPU"				#option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="MYRIAD"			#option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="HETERO:FPGA,CPU"		#option CPU,GPU,MYRIAD,HETERO:FPGA,CPU



im = Image.open(__image)					# Crop from image for openCV


#Step-3 ___Load plugin and cpu extension libraly for 2019R3 only
plugin = IEPlugin(device=__plug)
#if __plug=="CPU":                       #if plug is cpu , add extension for cpu only
#        plugin.add_cpu_extension("/home/ieisw/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so")

#Step-4 ___Set IENetwork with xml , bin
net = IENetwork(model=__xml,weights=__bin)

#Step-5 ___Set exec_net and load to network   
exec_net = plugin.load(network=net)



#Step-6 ___Read image and format for openvino format
img_face = cv2.imread(__image)
img = cv2.resize(img_face, (672,384))		   # for change format face-detection size must 672,384
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))    # HWC > CHW 
img = np.expand_dims(img, axis=0)


#Step-7 ___ execute inference engine with image and get result as "res" start sync mode
res = exec_net.infer(inputs={'data':img}) #infer progress
#___ execute inference engine with image and get result as res end


print("__Face detection Start___")
#print(res)		#print all result as res , thas is 1000 result 


#Step-8 ___'detection_out' is result from inference engine from res
res = res['detection_out']	
res = np.squeeze(res) 		

#Step-9 ___ Judge if res[2] is over 0.5 , append into values
values=[]
for index in range(len(res)):
	if res[index][2] > 0.5:
		values.append([str(index),res[index][2]])
		print(values[index])

		xmin = int(int(res[index][3] * img_face.shape[1]) * 1.0) #0.9
		ymin = int(int(res[index][4] * img_face.shape[0]) * 1.0) #1.3
		xmax = int(int(res[index][5] * img_face.shape[1]) * 1.0) #1.1
		ymax = int(int(res[index][6] * img_face.shape[0]) * 1.0) #1.3

		print("xmin=",xmin)
		print("ymin=",ymin)
		print("xmax=",xmax)
		print("ymax=",ymax)
                #Step-9-1 ___ print box for face in picture __
		cv2.rectangle(img_face, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)

                #Step-9-2 ___ Crop face in imag_face and save
		im_crop = im.crop((xmin, ymin, xmax, ymax))
		im_crop.save(str(index)+'crop'+'.png', quality=95)



#Step-10 ___show image with face box
cv2.imshow('img_face', img_face)	
cv2.waitKey(2000)
cv2.destroyAllWindows()
 

#Step-11 __show inference engine and device(plug)
print()
print("Used plugin device is",__plug)
print("Inference engine is",__xml)
print("__Face detection End____")

cv2.imwrite('result.png', img_face) 

