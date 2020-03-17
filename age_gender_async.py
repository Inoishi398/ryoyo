#__________________//////////////////////_______________
#__For Openvino execute inference engine with device age-gender
#__Written by Inoishi 2020/Mar/10 from scratch
#__OpenVino version 2019R3
#__Version 1.0(2020/Mar/10)
#__Version 1.1(2020/Mar/12)
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
from PIL import Image

import time



#___Init xml,bin and image for inference engine
ie = IECore()
__xml="/media/psf/Home/Documents/ubunts/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml"
__bin="/media/psf/Home/Documents/ubunts/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin"

#__xml="/home/intel/Documents/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml"
#__bin="/home/intel/Documents/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin"
#__xml="/home/ieisw/Documents/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml"
#__bin="/home/ieisw/Documents/intel/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin"

#__image="brad.png"
#__image="ange.png"
#__image="tange.png"
__image="0crop.png"
#__image="perfume.png"

#___Set device and plug for Inference engine cpu,gpu,myriad,fpga
__plug="CPU"                            #option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="GPU"                           #option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="MYRIAD"                        #option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="HETERO:FPGA,CPU"               #option CPU,GPU,MYRIAD,HETERO:FPGA,CPU


#__Set inference work mode sync or async	
#__mode="SYNC"		
__mode="ASYNC"



#___Load plugin and cpu extension libraly
plugin = IEPlugin(device=__plug)
#if __plug=="CPU":			#if plug is cpu , add extension for cpu only
#	plugin.add_cpu_extension("/home/ieisw/inference_engine_samples_build/intel64/Release/lib/libcpu_extension.so")

#___Set IENetwork with xml , bin
net = IENetwork(model=__xml,weights=__bin)


#___Set exec_net and load to network   
if __mode=="SYNC":
	exec_net = plugin.load(network=net)				#Sync mode
elif __mode=="ASYNC":
	exec_net = plugin.load(network=net, num_requests=2)		#Async mode


#___Read image and format for openvino format
img_face = cv2.imread(__image)
img = cv2.resize(img_face, (62,62))	
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))    # HWC > CHW 
img = np.expand_dims(img, axis=0)


#___ execute inference engine with image and get result as "res" start sync mode
if __mode=="SYNC":
	res = exec_net.infer(inputs={'data':img}) 			#Sync infer progress
elif __mode=="ASYNC":
	res = exec_net.start_async(request_id=0, inputs={'data': img})	#Astnc infer progress 
	if exec_net.requests[0].wait(-1) == 0:				#Async wait response
		res = exec_net.requests[0].outputs
	else:
		print("No response")

#___ Get result as res end
print("__Age-gender Start__")
#print("res__",res)		#print all result as res , thas is 1000 result 

#___Get result on res, key, deleate no space in array by squeeze
#___'age_conv3,prob' is result from inference engine from res
age=np.squeeze(res['age_conv3']*100)
prob=np.squeeze(res['prob'])

print("age_____",age)
print("female__",prob[0])
print("male____",prob[1])
print("image___",__image)
print()
print("plug__",__plug)
print("mode__",__mode)


#__show image with information
cv2.putText(img_face, ('age='+(str(age))) ,        (0,15),  cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1, cv2.LINE_AA)
cv2.putText(img_face, ('female='+(str(prob[0]))) , (0, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1, cv2.LINE_AA)
cv2.putText(img_face, ('male='+(str(prob[1]))) ,   (0, 45), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1, cv2.LINE_AA)

cv2.resize(img_face,dsize=(100, 200))
#cv2.imshow('result.png', img_face)
cv2.imwrite('img_face.png', img_face)
cv2.imshow('img_face.png', img_face)
cv2.waitKey(2000)
cv2.destroyAllWindows()



def result(age,female,male):

	result.age=np.round(age,decimals=2)
	result.female=np.round(female,decimals=2)
	result.male=np.round(male,decimals=2)

	print("Result_age==",result.age)
	print("Result_female==",result.female,"%")
	print("Result_male==",result.male,"%")

result(age,prob[0],prob[1])
print("__xml",__xml)
print("__Age-gender End____")

