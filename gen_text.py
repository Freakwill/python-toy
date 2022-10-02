#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from textgenrnn import textgenrnn

# PATH = pathlib.Path('~/Folders/mycorpus').expanduser()
# textgen = textgenrnn()  # 'textgenrnn_weights.hdf5'
# textgen.train_from_file(PATH / 'poem.txt', num_epochs=15)

textgen = textgenrnn('textgenrnn_weights.hdf5') 
textgen.generate()