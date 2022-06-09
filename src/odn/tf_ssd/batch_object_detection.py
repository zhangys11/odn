# %load batch_object_detection.py
# %load batch_object_detection.py
import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six

from distutils.version import StrictVersion
if StrictVersion(tf.__version__) < StrictVersion('1.4.0'):
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import importlib
importlib.reload(vis_util) # reflect changes in the source file immediately

from math import sqrt
from math import pow

from tqdm import tqdm


