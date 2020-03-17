#__________________//////////////////////_______________
#__For Openvino execute inference engine with all device
#__Squeezenet1.1 intel model
#__Written by Inoishi 2020/Mar/4 from scratch
#__OpenVino version 2019R3
#__Version 1.0 2020/Mar/10
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

#___Init xml,bin and image for inference engine
ie = IECore()
__xml="/home/intel/openvino_models/ir/public/squeezenet1.1/FP16/squeezenet1.1.xml"
__bin="/home/intel/openvino_models/ir/public/squeezenet1.1/FP16/squeezenet1.1.bin"
__label="squeezenet1.1.labels"
__image="cofee_cup.png"
#__image="video.png"

#___Set device and plug for Inference engine 
__plug="CPU"			#option CPU,GPU,MYRIAD,MULTI:**,HETERO:FPGA,CPU
#__plug="GPU"			
#__plug="MYRIAD"		
#__plug="HETERO:FPGA,CPU"	


#__Set a Enviroment for inference engine , net , input model , exec_net
#plugin = IEPlugin(device=__plug, plugin_dirs=None)
net = IENetwork(model=__xml,weights=__bin)
batch,channel,height,width = net.inputs['data'].shape
exec_net = ie.load_network(network=net, device_name=__plug, num_requests=1)

#___Read image and format for openvino format
img_face = cv2.imread(__image) #read image file
img = cv2.resize(img_face, (width,height)) 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose((2, 0, 1))
img = img.reshape((1, channel, height, width))

#img = cv2.imread(__image) #read image file
#img = cv2.resize(img, (width,height))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = img.transpose((2, 0, 1))
#img = img.reshape((1, channel, height, width))


#___ execute inference engine with image and get result as "res" start sync mode
res = exec_net.infer(inputs={'data':img}) 
#___ execute inference engine with image and get result as res end

#___Show all output from Inference engine for confirm result what did it result.
print("___Squeesenet1.1 Start")
print("__Start__")
print(res)		

#___for get item each in values , because res is array , all data in list in res
values=[]	
for item in res['prob'][0]:
        values.append(item[0][0])

sortedlist=np.sort(values)
list=[]	#make a list array,[0]=accracy, [1]=result cnt 0to999, [2]=labels from below

#___Read labels for squeezenet1.1 into lines as f
with open(__label) as f:	#open file as f
    lines = f.readlines() 	#read file as line

#___make a original list , include values(res),item(counter)
for item in range(0,1000): 
	list.append(["{:.5f}".format(values[item]*1),item,lines[item].splitlines()])
	print(list[item])



#___Show sort list best 10___
print("__sort list__")	
list.sort(key=itemgetter(0),reverse=True)	# sort by accuracy and reverse
for cnt in range(0,9):				# print best 10 with sorted list
	print(list[cnt])			# show best 10

#__show inference engine and device(plug)
print()
print("Used plugin device is",__plug)
print("Inference engine is",__xml)
print("___Squeesenet1.1 End")

cv2.imshow('img_face',img_face)
cv2.waitKey(2000)
cv2.destroyAllWindows()






