import numpy as np
from .keras_frcnn import roi_helpers
import cv2
import re
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import shutil
import os

def moving_average_3(a):    
	return np.concatenate(([a[0]], np.convolve(a, np.ones(3), 'valid') / 3, [a[-1]]))

def plot_training_curves(input_file = '../src/odn/training_log.txt', output_file = '../src/odn/training_curves.png'):
	'''
	Parameter
	---------
	input_file : the log file path. The log file is created from the commandline window output.
	'''

	L1={}
	L2={}
	L3={}
	L4={}
	ACC ={}

	epoch = 0

	with open(input_file) as f:
		content = f.readlines()
		for l in content:
			if l.startswith("Epoch"):
				arr = re.split('[: /]',l)
				epoch = int(arr[1])
			elif l.startswith("Classifier accuracy for bounding boxes from RPN"):
				arr = re.split('[:]',l)
				ACC[epoch] = float(arr[1])
			elif l.startswith("Loss RPN classifier"):
				arr = re.split('[:]',l)
				L1[epoch] = float(arr[1])
			elif l.startswith("Loss RPN regression"):
				arr = re.split('[:]',l)
				L2[epoch] = float(arr[1])
			elif l.startswith("Loss Detector classifier"):
				arr = re.split('[:]',l)
				L3[epoch] = float(arr[1])
			elif l.startswith("Loss Detector regression"):
				arr = re.split('[:]',l)
				L4[epoch] = float(arr[1])    

	fig, ax = plt.subplots(3,2, figsize = (40, 40))
	plt.rcParams.update({'font.size': 32})
	#plt.tick_params(axis='both', which='major', labelsize=24)
	#plt.tick_params(axis='both', which='minor', labelsize=24)
	plt.rcParams['xtick.labelsize'] = 32
	plt.rcParams['ytick.labelsize'] = 32

	labels = ['RPN Classification Error','RPN Regression Error','Region Classifier Classification Error',
			'Region Classifier Regression Error','Overall Classification Accuracy']

	M = [L1, L2, L3, L4, ACC]

	for idx in range(5):
		row = idx//2
		col = idx%2
		
		m = M[idx]
		
		ax[row, col].plot(list(m.keys()), moving_average_3(list(m.values())),label = labels[idx], color = 'gray', linewidth = 2)
		ax[row, col].scatter(list(m.keys()), moving_average_3(list(m.values())), s=240, facecolors='none', edgecolors='gray')
		ax[row, col].xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax[row, col].legend(loc='upper center')

	ax[2, 1].set_axis_off()

	plt.savefig(output_file)
	plt.close(fig)

def get_bbox(R, C, model_classifier, class_mapping, F, ratio, bbox_threshold = 0.8):
	# convert from (x1,y1,x2,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# apply the spatial pyramid pooling to the proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0]//C.num_rois + 1):
		ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			#pad R
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

			if cls_name not in bboxes:
				bboxes[cls_name] = []
				probs[cls_name] = []

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])
			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]
				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
			except:
				pass
			bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
			probs[cls_name].append(np.max(P_cls[0, ii, :]))
	all_dets = []
	for key in bboxes:
		bbox = np.array(bboxes[key])

		new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
		for jk in range(new_boxes.shape[0]):
			(x1, y1, x2, y2) = new_boxes[jk,:]

			(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

#			cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)
			textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
			all_dets.append((key,100*new_probs[jk]))
			(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
#			textOrg = (real_x1, real_y1-0)            
	return all_dets, bboxes, probs


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)


################ Below are used by tf-ssd ##################
import json
from PIL import Image
import imghdr
import matplotlib.patches as patches

def get_img_dims(path):
    if imghdr.what(path) is None:
        return None
    img = Image.open(path)
    return img.size


def parse_json_anno_file(path, imgdir):
    l = []
    with open(path, 'rb') as jf:                
        data = json.load(jf)
        for k,v in data.items():                    
            for p,q in v['regions'].items():
                new_item = {}
                new_item['filename'] = v['filename']
                for r,s in q['region_attributes'].items():
                    if ((s == 'OpticDisk' or s == 'Macula') and 'cx' in q['shape_attributes'].keys() and 'cy' in q['shape_attributes'].keys()):
                        imgfile = os.path.join(imgdir, new_item['filename'])
                        if os.path.isfile(imgfile):
                            dims = get_img_dims(imgfile)
                            if (dims is not None):
                                new_item['width'] = dims[0]
                                new_item['height'] = dims[1]
                                new_item['class'] = s
                                new_item['cx'] = q['shape_attributes']['cx']
                                new_item['cy'] = q['shape_attributes']['cy']

                if 'class' in new_item.keys():
                    # L001 - OD, L002 - OS
                    if 'labels8' in v['image_labels']:
                        new_item['laterality'] = v['image_labels']['labels8']
                    l.append(new_item)
                    
    return l
	
def visualize_bbox(img, xmin,ymin,xmax,ymax,text = ''):
    im = np.array(Image.open(img))
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    rc = patches.Rectangle((xmin, ymin),xmax-xmin,ymax-ymin,facecolor='none', edgecolor='b')
    ax.add_patch(rc)
    ax.annotate(text, (xmin, ymin), color='b', weight='bold', fontsize=12, ha='left', va='bottom')
    plt.show()
    
def visualize_bbox_for_anno_item(imgdir, item):
    '''
    Paramgers
	---------
	item : a dict object that contains filename, x and y coordinates
	'''
    if ('filename' not in item.keys() or 'xmin' not in item.keys() or 'xmax' not in item.keys() or 'ymin' not in item.keys() or 'ymax' not in item.keys()):
        print('Warning: item is invalid. It doesnot contain needed keys.')
        return
    imgfile = os.path.join(imgdir, item['filename'])
    if(imgfile is not None):
        visualize_bbox(imgfile, item['xmin'],item['ymin'],item['xmax'],item['ymax'],item['class'])

def visualize_bbox_center(img, cx, cy):
    im = np.array(Image.open(img))
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    c = patches.Circle((cx,cy),10, color = 'r')
    ax.add_patch(c)
    plt.show()

from PIL import Image

def convert_rgba_to_rgb(fname, folder):
    if (fname.endswith('.png')):
        jpgfile = os.path.join(folder, fname.replace('.png','.jpg'))
                
        png = Image.open(os.path.join(folder, fname))
        rgb_im = png.convert('RGB')
        rgb_im.save(jpgfile)
        
        #png.load() # required for png.split()
        #background = Image.new("RGB", png.size, (255, 255, 255))
        #background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
        #background.save(jpgfile, 'JPEG', quality=80)
        
        #png = Image.open(os.path.join(folder, fname)).convert('RGBA')
        #background = Image.new('RGBA', png.size, (255,255,255))
        #alpha_composite = Image.alpha_composite(background, png)
        #alpha_composite.save(jpgfile, 'JPEG', quality=80)
        os.remove(os.path.join(folder, fname))
        return jpgfile
    return None
        

import imghdr
def get_img_mode(path):
    if imghdr.what(path) is None:
        return None
    img = Image.open(path)    
    return img.mode

import os
def search_file(folder, fname):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file == fname:
                return (os.path.join(root, file))
    return None


def replace_image_in_list(l, file, newfile):
    for idx, item in enumerate(l):
        if(item['filename'] == file):
            item['filename'] = newfile
            print('R',  newfile)
        
def remove_image_in_list(l, file):
    for idx, item in enumerate(l):
        if(item['filename'] == file):
            l.pop(idx)
            print('D', file)
            return

def convert_to_rgb(folder, json_annos):
	'''
	确保所有图片为RGB模式，不是RGBA模式；将RGBA模式的PNG文件转换为JPG
	'''

	to_del = []
	to_replace = []

	for root, dirs, files in os.walk(folder):
			for file in files:
				if( not file.endswith('.db')):
					if (search_file(root, file) is not None):
						mode = get_img_mode(os.path.join(root, file))
						if mode is None:
							remove_image_in_list(json_annos, file)
							to_del.append(file)
						elif mode == 'RGBA' and file.endswith('png'): #there are also RGB mode png files!
							jf = convert_rgba_to_rgb(file, root)
							if(jf is not None):
								replace_image_in_list(json_annos, file, file.replace('.png','.jpg'))
								to_replace.append(file)
					else:
						print(file,' Not Found.')

def move_images_to_one_folder(source, target):
	'''
	Move all images (including subfolders) in the source folder to the target folder.
	'''
	os.makedirs(target, exist_ok=True)

	for root, dirs, files in os.walk(source):
		for file in files:
			if( not file.endswith('.db') and not file.endswith('.json')):
				shutil.move(os.path.join(root, file),os.path.join(target, file)) 


def merge_json_anno_files(source_folder, target_file):
	'''
	Merge all json anno files in the source folder to one single target json file.
	'''
	
	s = ''

	with open(target_file, 'w') as outfile:
		outfile.write('{')
		for root, dirs, files in os.walk(source_folder):
			for f in files:
				if f.endswith('.json'):
					with open(os.path.join(root, f), 'r') as jf:
						# outfile.write(jf.read()[1:-1])
						# outfile.write(',')
						s = s + jf.read()[1:-1] + ','
		s = s[:-1]
		outfile.write(s)
		outfile.write('}')   

def get_all_images_in_dir(folder = '../data/fundus/images_public/', target_file = None):

	FILES=[]	
	content = ""

	for root, dirs, files in os.walk(folder):
		for f in files:
			if( os.path.isfile(os.path.join(root, f)) and f.endswith('.jpg')):
				fp =  os.path.join(root, f).replace("\\", "/") 
				FILES.append(fp)
				content = content + fp + ",-1,-1,-1,-1,UNKNOWN" + "\n" # filename,x1,y1,x2,y2,class_name

	if target_file:
		file = open(target_file, 'w')
		file.write(content)
		file.close()

	return FILES