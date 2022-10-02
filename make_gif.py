#!/usr/local/bin/python
# -*- coding: utf-8 -*-

from moviepy.editor import *

#视频文件的本地路径
moviePath = 'template.mp4'
content = VideoFileClip(moviePath)

hm1 = 0, 37.8  # '9:41 - 9:52'
hm2 = 0, 49

size = (340, 220)
v = content.subclip(hm1, hm2).resize(size)

#v.write_gif("xiajian.gif")
v.without_audio()

texts =[
('我和她有共同语言', 500, 1500),
('放屁', 2000, 1000),
('你那是和她有共同语言吗？', 5200, 1500),
('你那是馋她身子 你下贱', 6500, 3000)]

textclips =[TextClip(text, 
    fontsize=60, color='white',
    bg_color='transparent',
    transparent=True
    ).set_position(('center', 'bottum')).set_duration(duration).set_start(start)
    for text, start, duration in texts]

v =  CompositeVideoClip([v]+textclips)
v.write_gif('xiajian.text.gif')
