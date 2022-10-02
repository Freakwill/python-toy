#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
To generate a jiugong (9-palace) viedo from any video
"""

import pathlib
from moviepy.editor import *
from PIL import Image

path_temp = pathlib.Path('temp')
path_temp.mkdir(exist_ok=True)

video_file ='piano.mp4'
video_raw_clip = VideoFileClip(video_file)
audio_clip = video_raw_clip.audio
video_width, video_height = video_raw_clip.w, video_raw_clip.h
fps = video_raw_clip.fps
during = video_raw_clip.duration

def make_frames(video_clip):
    item_space = 10
    space_color = (139, 0, 0)

    item_width = video_width // 3
    item_height = video_height // 3

    video_width_new = video_width + item_space * 2
    video_height_new = video_height + item_space * 2


    for k, frame in enumerate(video_clip.iter_frames(), start=1):
        image = Image.fromarray(frame)
        new_image = Image.new(image.mode, (video_width_new, video_height_new), color=space_color)
        for i in range(0, 3):
           for j in range(0, 3):
                box = (j * item_width, i * item_height, (j + 1) * item_width, (i + 1) * item_height)
                crop_image = image.crop(box)
                x = 0 if j == 0 else (item_width + item_space) * j
                y = 0 if i == 0 else (item_height + item_space) * i
                new_image.paste(crop_image, (x, y))
                new_image.save(path_temp / ("%04d.jpg" % k))

if True:
    make_frames(video_raw_clip)

images = filter(lambda f: f.suffix=='.jpg', path_temp.iterdir())
images = list(map(str, sorted(images, key=lambda p: int(p.stem))))

video_clip = ImageSequenceClip(images, fps=fps).set_audio(audio_clip)
video_clip.write_videofile('jiugong.mp4', fps=fps,
                            codec='libx264',
                            audio_codec='aac',
                            temp_audiofile='temp-audio.m4a',
                            remove_temp=True)

if False:
    for f in path_temp.iterdir():
        f.unlink()
    path_temp.rmdir()
