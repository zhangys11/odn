import os
from PIL import Image
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import numpy as np
def visualize_bbox(img, xmin,ymin,xmax,ymax,text = ''):
    im = np.array(Image.open(img))
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    rc = patches.Rectangle((xmin, ymin),xmax-xmin,ymax-ymin,facecolor='none', edgecolor='b')
    ax.add_patch(rc)
    ax.annotate(text, (xmin, ymin), color='b', weight='bold', fontsize=12, ha='left', va='bottom')
    plt.show()
    
def visualize_bbox_for_anno_item(imgdir, item):
    if ('filename' not in item.keys() or 'xmin' not in item.keys() or 'xmax' not in item.keys() or 'ymin' not in item.keys() or 'ymax' not in item.keys()):
        print('Warning: item is invalid. It doesnot contain needed keys.')
        return
    imgfile = os.path.join(imgdir, item['filename'])
    if(imgfile is not None):
        visualize_bbox(imgfile, item['xmin'],item['ymin'],item['xmax'],item['ymax'],item['class'])