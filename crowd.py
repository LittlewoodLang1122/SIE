## 本文件用于对人群进行建模
import numpy as np
from numpy import random as rdm
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

import utils

class crowd():
    """
        这个类的作用是模拟在没有采取任何措施的情况下，模拟传染病的过程



        使用一个四维数组代表一个人，其中前两个维度代表这个人的 x 坐标和 y 坐标
        第三个维度代表这个人的状态

            其中 
            0 代表 s，即易感人群
            1 代表 i，即感染者
            2 代表 r，即康复
            3 代表密接，4代表次密接
        第四个维度代表这个人的感染天数（如果他感染）
            s 的感染天数是 0
            r 的感染天数是 11（我们默认疾病10天内治愈）


        以上海市的规模为例（6400平方千米，2500万人），假定地图是正方形，（边长80*1000米），按比例缩小 80 倍，设定长宽为 1000，人口为 4000，来进行模型演化。
    """

    def __init__(self, size=4000, mod='dis') -> None:
        self.size = size
        self.mod = mod

        self.data = np.zeros((4, self.size), dtype=float)
        if self.mod == 'dis':
            self.data[0, :] = rdm.randint(-500, 500, self.size)
            self.data[1, :] = rdm.randint(-500, 500, self.size)

        if self.mod == 'con':
            self.data[0, :] = utils.normal_lu(self.size, lower=-500, upper=500, prop=3)
            self.data[1, :] = utils.normal_lu(self.size, lower=-500, upper=500, prop=3)

    def getLoc(self, index):
        return self.data[0:2, index]
    
    def initI(self, size):
        """
        设置初始的感染者数，因为位置是随机取的，所以直接设置前 （size） 人为初始的感染者，因为这一天他们不感染，所以不设定感染天数
        """
        self.data[2, 0:(size)] = 1

    def Foward(self, rate=0.5, dis=1.42):
        """
        在某天晚上进行清算，对当前位置的人判断是否感染，因为 I 只会感染 S 所以不考虑 R
        """
        idx_S = np.argwhere(self.data[2,:] == 0).squeeze()
        idx_I = np.argwhere(self.data[2,:] == 1).squeeze()

        ## 现在让 I 去感染 S
        for i in idx_I:
            ## 这里 i 是这个感染者的编号
            pos_i = self.getLoc(i)
            for s in idx_S:
                pos_s = self.getLoc(s)
                if utils.distance(pos_i, pos_s) >= dis:
                    ## 如果这两个人足够远则无事发生
                    continue
                else:
                    ## 否则看命
                    prob = rdm.rand(1)
                    if prob > rate:
                        ## 如果命好就无事发生
                        continue
                    else:
                        self.data[2,s] = 1

    def Move(self):
        """
        第二天每个人都开始走，同时感染者感染的天数+1
        """
        if self.mod == 'con':
            ## 连续情形下的移动，则随机分配位置
            self.data[0, :] = utils.normal_lu(self.size, lower=-500, upper=500, prop=3)
            self.data[1, :] = utils.normal_lu(self.size, lower=-500, upper=500, prop=3)
        if self.mod == 'dis':
            ## 离散情形下则随机走
            for i in range(self.size):
                para = rdm.randint(4)
                if para == 0:
                    self.data[0,i] += 1
                if para == 1:
                    self.data[0,i] -= 1
                if para == 2:
                    self.data[1,i] += 1
                if para == 3:
                    self.data[1,i] -= 1

        idx_I = np.argwhere(self.data[2,:] == 1).squeeze()
        for i in idx_I:
            ## 感染者感染天数加 1
            self.data[3, i] += 1
            if self.data[3, i] == 11:
                ## 出院！
                self.data[2, i] = 2
    
    def visualize(self):
        x = self.data[0,:]
        y = self.data[1,:]

        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        fig, ax = plt.subplots()
        plt.scatter(x, y,c=z, s=20,cmap='Spectral') # c表示标记的颜色
        plt.colorbar()

    def getStatusCrowd(self, status):
        """
        获取特定的人群
        """
        idx = np.argwhere(self.data[2,:] == status).squeeze()
        cwd = crowd(idx.shape[0], mod=self.mod)
        cwd.data = self.data[:, idx]

        return cwd