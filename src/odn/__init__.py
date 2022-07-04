
import pandas as pd
import os
import sys
if __package__:
    from . import utils, test_frcnn, fundus
else:
    sys.path.append(os.path.dirname(__file__))
    import utils, test_frcnn, fundus

def predict_folder(folder, method = 'FRCNN', dir_output = 'inplace'):
    '''
    method : 'FRCNN', 'SSD', or 'YOLO5'
    dir_output : specify a target folder or just 'inplace'
    '''
    
    wd = os.path.dirname(__file__) # + '/keras_frcnn/'
    fl = wd + '/filelist.tmp.txt' # uuid.uuid1() + '.txt'
    FILES = utils.get_all_images_in_dir(folder, fl)
    test_set = pd.read_csv(fl, header=None, encoding = 'gbk')
    test_frcnn.predict_images(fl, 'vgg16', wd + '/models/vgg16/19e.hdf5', 32,
    wd + '/candidate.rois.txt')

    rois = fundus.annotation.rule_filter_rois(input_file = wd + '/candidate.rois.txt', 
    output_file = wd + '/filtered.rois.txt',
    verbose = False)

    fundus.dataset.synthesize_anno(wd + '/filtered.rois.txt', 
                            dir_images = folder, 
                            dir_output = dir_output,
                            drawzones = True,
                            verbose = False,
                            suffix = '_FRCNN',
                            display = 0 )

predict_folder('C:/Users/eleve/Desktop/横向2022/SZEH分区 第二次激光/激光  85个第二次', dir_output = 'inplace')
