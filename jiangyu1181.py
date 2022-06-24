# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Jiangyu1181
# Github: https://github.com/Jiangyu1181
# ----------------------------------------------------------------------------------------------------------------------
# Description: ""
# Reference Paper: ""
# Reference Code: ""
# ----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import animation


def img_plt(img):
    img = plt.imread("/mnt/data2/jy/datasets/nerfactor/rendered-images/human/test_%03d/rgba.png" % img)
    img_plt_ = plt.imshow(img)
    ax = plt.gca()
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()
    return img_plt_


dpi = 96
width, height = 512 / dpi, 512 / dpi
anim = animation.ArtistAnimation(
    plt.figure(figsize=(width, height)), [(img_plt(x),) for x in range(200)], interval=200)
anim.save("/mnt/data2/jy/datasets/nerfactor/rendered-images/%s.gif" % "human", writer='pillow')
