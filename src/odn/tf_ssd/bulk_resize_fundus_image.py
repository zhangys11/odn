import os
import sys
from PIL import Image

# All fundus images are 4:3 ratio. 
def resize_fundus_image(root, f, target, prefix='', w=480, h=360):
    filePath = os.path.join(root, f)
    newfilePath = os.path.join(target, prefix + f)    
    im = Image.open(filePath).resize((w,h)) 
    im.save(newfilePath)

# NOTE: images in the target folder will be overwritten if prefix is empty
def bulk_resize_fundus_image(folder, target, prefix = '', w=480, h=360):
    os.makedirs(target, exist_ok=True)
    
    imgExts = ["png", "bmp", "jpg"]
    for root, dirs, files in os.walk(folder):
        for f in files:
            ext = f[-3:].lower()
            if ext not in imgExts:
                continue
            if prefix!='' and prefix is not None and f.startswith(prefix):
                continue
            resize_fundus_image(root, f, target = target, prefix=prefix, w = w, h = h)