import unittest
import numpy as np
from utils.visualize import __read_vox__, __read_vox_frag__, plot, plot_frag, plot_join
import pyvox.parser

if __name__ == '__main__':
    vox1 = __read_vox__("./data/test1.vox")
    vox2 = __read_vox__("./data/test2.vox")
    vox1_1 = __read_vox_frag__("./data/test1.vox", 1)
    vox2_1 = __read_vox_frag__("./data/test2.vox", 1)
    plot(vox1, "./figures")
    plot(vox2, "./figures")
    plot_frag(vox1, "./figures")
    plot_frag(vox2, "./figures")
    plot_join(vox1, vox2, "./figures")

# python unittest.py