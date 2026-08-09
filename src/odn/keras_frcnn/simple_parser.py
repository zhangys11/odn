from unittest import skip
import cv2
import numpy as np

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
	cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
	# imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
	# cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
	return cv_img

def get_data(input_path, cat = None, skip_header = False, path_prefix = ""):

	found_bg = False
	all_imgs = {}

	classes_count = {}

	class_mapping = {}

	visualise = True
	
	with open(input_path,'r') as f:

		print('Parsing annotation files')

		for line in f:
			if skip_header:
				skip_header = False
				continue # skip 1st row

			line_split = line.strip().split(',')
			(filename,x1,y1,x2,y2,class_name) = line_split

			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:
				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True
				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}
				# print(path_prefix + filename)
				img = cv_imread(path_prefix + filename) # cv2.imread(path_prefix + filename)

				(rows,cols) = img.shape[:2]
				# print(rows, cols)

				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['bboxes'] = []
				if np.random.randint(0,6) > 0:
					all_imgs[filename]['imageset'] = 'trainval'
				else:
					all_imgs[filename]['imageset'] = 'test'

			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': round(float(x1)), 'x2': round(float(x2)), 'y1': round(float(y1)), 'y2': round(float(y2))})

		all_data = []
		for key in all_imgs:
			all_data.append(all_imgs[key])
		
		# make sure the bg class is last in the list
		if found_bg:
			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
		
		return all_data, classes_count, class_mapping


