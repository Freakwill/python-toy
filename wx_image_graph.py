#!/usr/local/bin/python
# -*- coding: utf-8 -*-


"""load the images in wechat

requirements:
    Pillow
    PIL_ext (extension of pillow)
    numpy
    itchat
"""

import pathlib
import copy
import PIL.Image as Image

import numpy as np

from PIL_ext import *

def make_heart(imgs, path=None):
    """make a heart from the images
    imgs: images
    path: the path where the output image is stored
    """

    def inside(point):
        # Wheher the point in the area of the heart❤️
        if point == (0, 0):
            return True
        else:
            x, y = point
            if 16 >= x >= 0:        
                if y >= 0:
                    t = np.arcsin((x / 16) ** (1/3))
                    return y <= np.round(13*np.cos(t) - 5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))
                else:
                    t = np.pi - np.arcsin((x /16) ** (1/3))
                    return y >= np.round(13*np.cos(t) - 5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t))
            elif x > 16:
                return False
            else:
                return inside((-x, y))

    nrow = 15 * 2
    ncol = 18 * 2

    bg = Background(nrow, ncol, 60)  # 先生成头像集模板

    coords = []

    for x in range(ncol):
        for y in range(nrow):
            point = (x-ncol//2 + 1, nrow//2-y - 2)
            if inside(point):
                coords.append((x,y))

    np.random.shuffle(imgs)
    rest = copy.deepcopy(coords)
    for img, coord in zip(itertools.cycle(imgs), coords):
        # 将图片循环平铺到心形区域中，随机设定大小
        if coord in rest:
            x, y = coord
            s = 1
            if y >= 15:
                s = np.random.randint(2,8)
                if (x +s-1, y) in rest and (x +s-1, y+s-1) in rest and (x, y+s-1) in rest:
                    for xi in range(x,x+s):
                        for yi in range(y,y+s):
                            if (xi, yi) in rest:
                                rest.remove((xi, yi))
                    bg.paste(img, coord, scale=s)
                else:
                    bg.paste(img, coord)
            elif x<18:
                s = np.random.randint(2, 5)
                if (x +s-1, y) in rest and (x +s-1, y+s-1) in rest and (x, y+s-1) in rest and x +s-1 < 18:
                    for xi in range(x,x+s):
                        for yi in range(y,y+s):
                            if (xi, yi) in rest:
                                rest.remove((xi, yi))
                    bg.paste(img, coord, scale=s)
                else:
                    bg.paste(img, coord)
            else:
                s = np.random.randint(2, 5)
                if (x +s-1, y) in rest and (x +s-1, y+s-1) in rest and (x, y+s-1) in rest and x +s-1 >= 18:
                    for xi in range(x,x+s):
                        for yi in range(y,y+s):
                            if (xi, yi) in rest:
                                rest.remove((xi, yi))
                    bg.paste(img, coord, scale=s)
                else:
                    bg.paste(img, coord)
    if path:
        bg.save(path)
    else:
        bg.save("heart.jpg")

def load_wx_images(path=None, to_heart=False):
    # 下载所有微信头像，保存到path下
    # 多线程下载，体验闪电般的速度
    # path: 文件夹路径
    # to_heart: 是否做一个心形
    
    import itchat
    import threading

    itchat.auto_login(hotReload=True)
    friends = itchat.get_friends(update=True)   # 核心：得到friends列表集，内含很多信息
    print(friends)

    path = pathlib.Path(path) if path else pathlib.Path('images')
    path.mkdir(parents=True, exist_ok=True)

    threads = []
    N = len(friends)
    for i, f in enumerate(friends):
        print(f"saving {i}/{N}")
        target = lambda i, f: (path / ("%d.jpg" % i)).write_bytes(itchat.get_head_img(userName=f["UserName"]))
        threads.append(threading.Thread(target=target, args=(i, f)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if to_heart:
        imgs = []
        for f in path.iterdir():
            if f.suffix == '.jpg':
                try:
                    im = Image.open(f)
                    imgs.append(im)
                except:
                    pass
        make_heart(imgs)

if __name__ == '__main__':
    load_wx_images()
    
    