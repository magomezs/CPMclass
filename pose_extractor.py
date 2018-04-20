import caffe
import numpy as np
import os
import sys
import io
import cv2 as cv 
import scipy
import PIL.Image
import math
import time
from config_reader import config_reader
import util
import copy
import math

#sys.path.insert(0,'/home/magomezs/caffe-master' + '/python')


class PoseExtractor(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 1:
       raise Exception('must have exactly one input')
    if len(top) != 9:
       raise Exception('must have exactly one output')
    self.param, self.model = config_reader()
    boxsize = self.model['boxsize']
    npart = self.model['np']
    if self.param['use_gpu']:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    caffe.set_device(self.param['GPUdeviceNumber']) # set to your device! 
    self.pose_net = caffe.Net(self.model['deployFile'], self.model['caffemodel'], caffe.TEST) 
    #pose_net.blobs['image'].reshape(*(1,3,128,64))
    #person_net.reshape()
    self.pose_net.forward() # dry run to avoid GPU synchronization later in caffe
    self.factor=2
    self.batch=bottom[0].shape[0]
     #print batch
    self.chanels=bottom[0].shape[1]
     #print chanels
    self.height= bottom[0].shape[2]*self.factor
     #print height 
    self.width= bottom[0].shape[3]*self.factor
    #print "L"


 

  def reshape(self,bottom,top):
       top[0].reshape(self.batch, 3, 45, 45)
       top[1].reshape(self.batch, 3, 40, 30)
       top[2].reshape(self.batch, 3, 40, 30)
       top[3].reshape(self.batch, 3, 40, 30)
       top[4].reshape(self.batch, 3, 40, 30)
       top[5].reshape(self.batch, 3, 60, 30)
       top[6].reshape(self.batch, 3, 60, 30)
       top[7].reshape(self.batch, 3, 60, 30)
       top[8].reshape(self.batch, 3, 60, 30)

        # check input dimensions match
        # if bottom[0].count != bottom[1].count:
        #    raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
       # self.diff = np.zeros(bottom[0].num, dtype=np.float32)
       # self.dist_sq = np.zeros(bottom[0].num, dtype=np.float32)
       # self.dist_norm = np.zeros(bottom[0].num, dtype=np.float32)
       # self.zeros = np.zeros(bottom[0].num)
       # self.ones =np.ones(bottom[0].num)
        # loss output is scalar
        #top[0].reshape(1)

   
  def forward(self, bottom, top):
     
     input_image = np.zeros((128, 64, 3, self.batch))
     for p in range(self.batch):
	    for x_p in range(64):
		for y_p in range(128):
		    if x_p >= 0 and x_p < 64 and y_p >= 0 and y_p < 128:
		       input_image[y_p, x_p, :, p] = np.float32(bottom[0].data[p,:, y_p, x_p])/255
     
     
     input_image_resize=np.ones((128*2, 64*2, 3, self.batch))/2
     for p in range(self.batch):
       image = np.zeros((128, 64, 3))
       image=input_image[:,:,:,p].copy()
       input_image_resize[:,:,:,p] = cv.resize(image, (0,0), fx=2, fy=2, interpolation = cv.INTER_CUBIC)
                


     person_image=np.ones((self.model['boxsize'], self.model['boxsize'], 3, self.batch))/2#person_image = np.ones((128, 64, 3, batch))/2#
     for p in range(self.batch):
	    for x_i in range(self.width):
		for y_i in range(self.height):
		    x_p = self.model['boxsize']/2 - self.width/2 + x_i
		    y_p = self.model['boxsize']/2 -self.height/2 + y_i
		    #if x_p >= 0 and x_p < width and y_p >= 0 and y_p < height:
		    person_image[y_p, x_p, :, p] = input_image_resize[y_i, x_i, :, p]#np.float32(bottom[0].data[p,:, y_p, x_p])/255
                       

     #cv.imshow( "gh", person_image[:,:,:,0])
     #cv.waitKey()
     #cv.imshow( "gh", person_image[:,:,:,7])
     #cv.waitKey()
     #cv.imshow( "gh", person_image[:,:,:,39])
     #cv.waitKey()


     gaussian_map = np.zeros((self.model['boxsize'], self.model['boxsize']))#gaussian_map = np.zeros((128, 64))#
     for x_p in range(self.model['boxsize']):
	    for y_p in range(self.model['boxsize']):
		dist_sq = (x_p - self.model['boxsize']/2) * (x_p - self.model['boxsize']/2) + (y_p - self.model['boxsize']/2) * (y_p - self.model['boxsize']/2)
		exponent = dist_sq / 2.0 / self.model['sigma'] / self.model['sigma']
		gaussian_map[y_p, x_p] = math.exp(-exponent)
     #cv.imshow("gau", gaussian_map)
     #cv.waitKey()

#7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777

     #pose_net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST) 
      #pose_net.blobs['image'].reshape(*(1,3,128,64))
      #person_net.reshape()
     #pose_net.forward() # dry run to avoid GPU synchronization later in caffe
     output_blobs_array = [dict() for dummy in range(bottom[0].shape[0])]
     for p in range(self.batch):
        input_4ch = np.ones((self.model['boxsize'], self.model['boxsize'], 4))#input_4ch = np.ones((128, 64, 4))#
        input_4ch[:,:,0:3] = person_image[:,:,:,p] - 0.5 # normalize to [-0.5, 0.5]
        input_4ch[:,:,3] = gaussian_map
        self.pose_net.blobs['data'].data[...] = np.transpose(np.float32(input_4ch[:,:,:,np.newaxis]), (3,2,0,1))
        #start_time = time.time()
        output_blobs_array[p] = copy.deepcopy(self.pose_net.forward()['Mconv7_stage6'])
        # print('For person %d, pose net took %.2f ms.' % (p, 1000 * (time.time() - start_time)))

     #print "aaa"
     #for p in range(self.batch):
	   # print('Person %d' % p)
	#    down_scaled_image = cv.resize(person_image[:,:,:,p], (0,0), fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
	 #   canvas = np.empty(shape=(self.model['boxsize']/2, 0, 3))
	  #  for part in [0,3,7,10,12]: # sample 5 body parts: [head, right elbow, left wrist, right ankle, left knee]
	#	part_map = output_blobs_array[p][0,part,:,:]
	#	part_map_resized = cv.resize(part_map, (0,0), fx=4, fy=4, interpolation=cv.INTER_CUBIC) #only for displaying
	#	part_map_color = util.colorize(part_map_resized)
	#	part_map_color_blend = np.float32(part_map_color)/255.0 * 0.5 + np.float32(down_scaled_image) * 0.5
	#	canvas = np.concatenate((canvas, part_map_color_blend), axis=1)
	#	canvas = np.concatenate((canvas, np.ones((self.model['boxsize']/2, 5, 3))), axis=1)

     #print "bbb"
     prediction = np.zeros((14, 2, self.batch))
     for p in range(self.batch):
	    for part in range(14):
		part_map = output_blobs_array[p][0, part, :, :]
		part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
		prediction[part,:,p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
	    # mapped back on full image
	    prediction[:,0,p] = prediction[:,0,p] #- (model['boxsize']/2) + y[p]
	    prediction[:,1,p] = prediction[:,1,p] #- (model['boxsize']/2) + x[p]


     limbs = self.model['limbs']
     stickwidth = 6
     colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],[255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255]] # note BGR ...
     #canvas = np.float32(imageToTest.copy())/255
     #cv.imshow("a", canvas)

    # cur_canvas = canvas.copy()*255



     head_im = np.zeros((45, 45, 3, self.batch))
     asi_im = np.zeros((40, 30, 3, self.batch))
     asd_im = np.zeros((40, 30, 3, self.batch))
     aii_im = np.zeros((40, 30, 3, self.batch))
     aid_im = np.zeros((40, 30, 3, self.batch))
     lsi_im = np.zeros((60, 30, 3, self.batch))
     lsd_im = np.zeros((60, 30, 3, self.batch))
     lii_im = np.zeros((60, 30, 3, self.batch))
     lid_im = np.zeros((60, 30, 3, self.batch))


     #result_image = np.zeros((model['boxsize'], model['boxsize'], 3, batch))
     for p in range(self.batch):
	    individual_image = np.zeros((self.model['boxsize'], self.model['boxsize'], 3))
            individual_image = person_image[:,:,:,p].copy()
            #cv.circle(dst, (int(point_r[1]), int(point_r[0])), 3, (255, 255, 255), -1)
            #cv.imshow("mjgko", individual_image)
            #cv.waitKey()
           
	    for l in range(limbs.shape[0]):
		X = prediction[limbs[l,:]-1, 0, p]
		Y = prediction[limbs[l,:]-1, 1, p]
		#np.savetxt("parts.txt", l.shape[0])
                #np.savetxt("parts.txt", limbs[l,:]-1)
		mX = np.mean(X)
		mY = np.mean(Y)
		length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
		#polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                #cv.circle(individual_image, (int(Y[0]), int(X[0])), 3, (255, 0, 255), -1)
		#cv.fillConvexPoly(individual_image, polygon, colors[l])
                #cv.imshow("result", individual_image)
                #cv.waitKey()


                #imagen original rotada
                M = cv.getRotationMatrix2D((self.model['boxsize']/2,self.model['boxsize']/2),angle+90,1)
                dst = cv.warpAffine(individual_image,M,(self.model['boxsize'],self.model['boxsize']))
                v = [int(X[0]),int(Y[0])]
                point_r=rotate((self.model['boxsize']/2,self.model['boxsize']/2),v,math.radians(angle+90))#np.dot(rotation_matrix((model['boxsize']/2,model['boxsize']/2),angle+90), v)
                #cv.circle(dst, (int(point_r[1]), int(point_r[0])), 3, (255, 255, 255), -1)
                #cv.imshow("rotated", dst)
                #cv.waitKey()

                #extraccion de la extremidad
                if l==0:
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(head_im.shape[0])
                   x_min=int(point_r[1])-int(head_im.shape[1]/2)-1
                   x_max=int(point_r[1])+int(head_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   head_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "0", (1,10), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey()
		   for x_p in range(head_im.shape[1]):
			for y_p in range(head_im.shape[0]):
			    if x_p >= 0 and x_p <head_im.shape[1]  and y_p >= 0 and y_p < head_im.shape[0]:
			       top[0].data[p,:, y_p, x_p]=head_im[y_p, x_p, :, p]
                if l==1 :
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(asd_im.shape[0])
                   x_min=int(point_r[1])-int(asd_im.shape[1]/2)
                   x_max=int(point_r[1])+int(asd_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   asd_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "2", (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey()  
		   for x_p in range(asd_im.shape[1]):
			for y_p in range(asd_im.shape[0]):
			    if x_p >= 0 and x_p <asd_im.shape[1]  and y_p >= 0 and y_p < asd_im.shape[0]:
			       top[1].data[p,:, y_p, x_p]=asd_im[y_p, x_p, :, p]

		if l==2:
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(aid_im.shape[0])
                   x_min=int(point_r[1])-int(aid_im.shape[1]/2)
                   x_max=int(point_r[1])+int(aid_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   aid_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "4", (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey()
		   for x_p in range(aid_im.shape[1]):
			for y_p in range(aid_im.shape[0]):
			    if x_p >= 0 and x_p <aid_im.shape[1]  and y_p >= 0 and y_p < aid_im.shape[0]:
			       top[2].data[p,:, y_p, x_p]=aid_im[y_p, x_p, :, p]
		   
		if l==3 :
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(asi_im.shape[0])
                   x_min=int(point_r[1])-int(asi_im.shape[1]/2)
                   x_max=int(point_r[1])+int(asi_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   asi_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "1", (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey() 
		   for x_p in range(asi_im.shape[1]):
			for y_p in range(asi_im.shape[0]):
			    if x_p >= 0 and x_p <asi_im.shape[1]  and y_p >= 0 and y_p < asi_im.shape[0]:
			       top[3].data[p,:, y_p, x_p]=asi_im[y_p, x_p, :, p]
                if l==4 :
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(aii_im.shape[0])
                   x_min=int(point_r[1])-int(aii_im.shape[1]/2)
                   x_max=int(point_r[1])+int(aii_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   aii_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "3", (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey()  
		   for x_p in range(aii_im.shape[1]):
			for y_p in range(aii_im.shape[0]):
			    if x_p >= 0 and x_p <aii_im.shape[1]  and y_p >= 0 and y_p < aii_im.shape[0]:
			       top[4].data[p,:, y_p, x_p]=aii_im[y_p, x_p, :, p]
                if l==5 :
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(lsd_im.shape[0])
                   x_min=int(point_r[1])-int(lsd_im.shape[1]/2)
                   x_max=int(point_r[1])+int(lsd_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   lsd_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "6", (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey()  
		   for x_p in range(lsd_im.shape[1]):
			for y_p in range(lsd_im.shape[0]):
			    if x_p >= 0 and x_p <lsd_im.shape[1]  and y_p >= 0 and y_p < lsd_im.shape[0]:
			       top[5].data[p,:, y_p, x_p]=lsd_im[y_p, x_p, :, p]
                if l==6 :
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(lid_im.shape[0])
                   x_min=int(point_r[1])-int(lid_im.shape[1]/2)
                   x_max=int(point_r[1])+int(lid_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   lid_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "8", (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey()  
		   for x_p in range(lid_im.shape[1]):
			for y_p in range(lid_im.shape[0]):
			    if x_p >= 0 and x_p <lid_im.shape[1]  and y_p >= 0 and y_p < lid_im.shape[0]:
			       top[6].data[p,:, y_p, x_p]=lid_im[y_p, x_p, :, p]
                if l==7 :
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(lsi_im.shape[0])
                   x_min=int(point_r[1])-int(lsi_im.shape[1]/2)
                   x_max=int(point_r[1])+int(lsi_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   lsi_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "5", (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey()  
		   for x_p in range(lsi_im.shape[1]):
			for y_p in range(lsi_im.shape[0]):
			    if x_p >= 0 and x_p <lsi_im.shape[1]  and y_p >= 0 and y_p < lsi_im.shape[0]:
			       top[7].data[p,:, y_p, x_p]=lsi_im[y_p, x_p, :, p]
                if l==8 :
                   y_min=int(point_r[0])
                   y_max=int(point_r[0])+int(lii_im.shape[0])
                   x_min=int(point_r[1])-int(lii_im.shape[1]/2)
                   x_max=int(point_r[1])+int(lii_im.shape[1]/2)
                   roi=dst[y_min:y_max, x_min:x_max]
                   lii_im[:,:,:,p]=np.float32(roi)
		   #cv.putText(roi, "7", (1,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, 255)
		   #cv.imshow("part", roi)
                   #cv.waitKey()  
		   for x_p in range(lii_im.shape[1]):
			for y_p in range(lii_im.shape[0]):
			    if x_p >= 0 and x_p <lii_im.shape[1]  and y_p >= 0 and y_p < lii_im.shape[0]:
			       top[8].data[p,:, y_p, x_p]=lii_im[y_p, x_p, :, p]







            #result_image[:,:,:,p]=individual_image.copy()
  
    # print np.amin(cur_canvas)
    # print np.amax(cur_canvas)
    # print np.amin(canvas)
    # print np.amax(canvas)
    # cur_canvas=np.float32(cur_canvas)/255.0
    # canvas = np.float32(canvas) * 0.4 + np.float32(cur_canvas) * 0.6  # for transparency
    # print np.amin(canvas)
    # print np.amax(canvas)
    # cv.imshow("9", canvas)
    # cv.waitKey()
     del head_im 
     del asi_im 
     del asd_im 
     del aii_im 
     del aid_im 
     del lsi_im 
     del lsd_im 
     del lii_im 
     del lid_im 
     del input_image
     del input_image_resize
     del person_image
     del gaussian_map 
     del x_p 
     del y_p  
     del x_i 
     del y_i
     del dist_sq 
     del exponent 
     del output_blobs_array 
     del input_4ch 
     del prediction 
     del p
     del part 
     del part_map 
     del part_map_resized 
     del limbs
     del stickwidth
     del colors
     del y_min
     del y_max
     del x_min
     del x_max
     del roi
     del dst
     del individual_image
     del l 
     del X 
     del Y 
     del mX
     del mY 
     del length 
     del angle 
     del M 
     del v
     del point_r


    
  def backward(self, top, propagate_down, bottom):
        # no back prop
      pass



def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)

