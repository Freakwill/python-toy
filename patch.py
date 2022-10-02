#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import PIL_ext

from PIL import *

p = pathlib.Path('test')
images = PIL_ext.get_images(folder=p)
big = Image.open('candy.jpg')
big=big.resize((50,50))
image = PIL_ext.patch(images, big)
image.save('candyx.jpg')
   