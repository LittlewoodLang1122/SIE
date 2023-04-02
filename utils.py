import numpy as np
import numpy.random as rdm
from d2l import torch as d2l
from crowd import crowd
from crowd_E import crowd_E


def normal_lu(size, lower, upper, prop=4):
    """
    生成一个在 lower 和 upper 之间的正态分布数组，以prop倍方差舍去外围的异常值，这样会使边缘有一些异常但是概率很小可以忽略
    """
    mean = (lower + upper) / 2
    sd = (upper - lower) / 2 / prop

    a = rdm.normal(mean, sd, size)
    a[a > upper] = upper
    a[a < lower] = lower
    return a

def distance(p1, p2):
    """
    测量两个人之间的距离，p1 和 p2 分别指两个人的位置
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def Process(c:'crowd|crowd_E', days:int=50, init=100, dis=2.82, rate=0.5, animator=False):
    data_container = []
    if isinstance(c, crowd):
        legend = ['S', 'I', 'R']
    else:
        legend = ['S', 'I', 'R', 'E']

    c.initI(init)
    for i in range(days):
        print(i)
        c.Move()
        c.Forward(rate=rate, dis=dis)
        data_container.append(c.getData())


    return data_container