import numpy as np
import numpy.random as rdm


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