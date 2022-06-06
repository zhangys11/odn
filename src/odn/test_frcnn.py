from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras.applications.mobilenet import preprocess_input

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", type="int", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.", default='resnet50')
parser.add_option("--write", dest="write", help="to write out the image with detections or not.", action='store_true')
parser.add_option("--load", dest="load", help="specify model path.", default=None)
(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# we will use resnet. may change to vgg
if options.network == 'vgg' or options.network == 'vgg16':
	C.network = 'vgg16'
	from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
	from keras_frcnn import resnet as nn
	C.network = 'resnet50'
elif options.network == 'vgg19':
	from keras_frcnn import vgg19 as nn
	C.network = 'vgg19'
elif options.network == 'mobilenetv1':
	from keras_frcnn import mobilenetv1 as nn
	C.network = 'mobilenetv1'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv1_05':
	from keras_frcnn import mobilenetv1_05 as nn
	C.network = 'mobilenetv1_05'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv1_25':
	from keras_frcnn import mobilenetv1_25 as nn
	C.network = 'mobilenetv1_25'
#	from keras.applications.mobilenet import preprocess_input
elif options.network == 'mobilenetv2':
	from keras_frcnn import mobilenetv2 as nn
	C.network = 'mobilenetv2'
else:
	print('Not a valid model')
	raise ValueError

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
		
	if width <= height:
		ratio = img_min_side/width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side/height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)

class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if C.network == 'resnet50':
	num_features = 1024
elif C.network =="mobilenetv2":
	num_features = 320
else:
	# may need to fix this up with your backbone..!
	print("backbone is not resnet50. number of features chosen is 512")
	num_features = 512

if K.image_data_format() == 'channels_first': # K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (num_features, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, num_features)


img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))

model_rpn = Model(img_input, rpn_layers)
model_classifier = Model([feature_map_input, roi_input], classifier)

from keras.utils import plot_model
plot_model(model_rpn, to_file='model_rpn.png', show_shapes = True)
plot_model(model_classifier, to_file='model_classifier.png', show_shapes = True)
#model_all = Model([img_input, roi_input], model_rpn[:2] + model_classifier)
#plot_model(model_all, to_file='model_all.png', show_shapes = True)


# model loading
if options.load == None:
  print('Loading weights from {}'.format(C.model_path))
  model_rpn.load_weights(C.model_path, by_name=True)
  model_classifier.load_weights(C.model_path, by_name=True)
else:
  print('Loading weights from {}'.format(options.load))
  model_rpn.load_weights(options.load, by_name=True)
  model_classifier.load_weights(options.load, by_name=True)

#model_rpn.compile(optimizer='adam', loss='mse')
#model_classifier.compile(optimizer='adam', loss='mse')

# classes = {}
# bbox_threshold = 0.5
# visualise = True

num_rois = C.num_rois

# if options.parser == 'simple':
from keras_frcnn.simple_parser import get_data
all_imgs, _, _ = get_data(img_path)

unique_imgs = []

import pandas as pd
candidate_rois = pd.DataFrame(columns=['filename', 'class', 'prob', 'width','height', 'xmin', 'ymin', 'xmax', 'ymax'])
candidate_rois.index.name = 'seq'

idx = 0

for img_info in all_imgs: # sorted(os.listdir(img_path))):
	img_name = img_info['filepath']
	if (img_name in unique_imgs):
		continue; # skip duplicatet images
	else:
		unique_imgs.append(img_name)
	print (img_name)
	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
		continue
	
	st = time.time()
	# filepath = os.path.join(img_path,img_name)

	img = cv2.imread(img_name)

    # preprocess image
	X, ratio = format_img(img, C)
	img_scaled = (np.transpose(X[0,:,:,:],(1,2,0)) + 127.5).astype('uint8')
	if K.image_data_format() == 'channels_last': # K.image_dim_ordering() == 'tf':
		X = np.transpose(X, (0, 2, 3, 1))
	# get the feature maps and output from the RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	# print('Y1', Y1.shape, Y1)
	# print('Y2', Y2.shape, Y2)
	# print('F', F.shape, F)

	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.5)    
    # overlap threshold for non_max_suppression algorithm
    
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]
	# print('Proposed Regions:', R.shape, R)

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}
	for jk in range(R.shape[0]//num_rois + 1):
		ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1),:],axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:,:curr_shape[1],:] = ROIs
			ROIs_padded[0,curr_shape[1]:,:] = ROIs[0,0,:]
			ROIs = ROIs_padded

		[P_cls,P_regr] = model_classifier.predict([F, ROIs])
		# print('P_cls', P_cls.shape, P_cls) # P_cls is (1, number of maximum rois, probabilities of each class)

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0,ii,:]) < 0.8 or np.argmax(P_cls[0,ii,:]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0,ii,:])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []
			(x,y,w,h) = ROIs[0,ii,:]

			bboxes[cls_name].append([16*x,16*y,16*(x+w),16*(y+h)])
			probs[cls_name].append(np.max(P_cls[0,ii,:]))

	# print('bboxes', bboxes)
	# print('probs', probs)

	all_dets = []

	for key in bboxes:
		#print(key)
		#print(len(bboxes[key]))
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh = 0.2, max_boxes = 10)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]
			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

			(height,width,_) = img.shape
			textLabel = '{}: {}%'.format(key,int(100*new_probs[jk]))
			candidate_rois.loc[len(candidate_rois)] = [os.path.basename(img_name), 
                          key, new_probs[jk], width, height, real_x1, real_y1, real_x2, real_y2]
			idx += 1
			all_dets.append((key,100*new_probs[jk])) 

			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.8,1)
			textOrg = (real_x1, real_y1-0)

			cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
			cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
			cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)

	# print('Elapsed time = {}'.format(time.time() - st))
	# print('all_dets', all_dets) 
	# # all_dets [('OpticDisk', 99.58785772323608), ('Macula', 89.3084704875946), ('Macula', 87.04832196235657)]

    # enable if you want to show pics
	if options.write:
           import os
           if not os.path.isdir("results"):
              os.mkdir("results")
           cv2.imwrite('./results/'+ os.path.basename(img_name),img)

	# break; # test the first image
	#candidate_rois[os.path.basename(img_name)] = all_dets 

# import pickle
# pickle.dump(candidate_rois, open('candidate_rois.pickle', 'wb'))
candidate_rois.to_csv('candidate_rois.txt')