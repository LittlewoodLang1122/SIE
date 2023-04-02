import crowd
import utils
import numpy as np
from numpy import random as rdm

class crowd_E(crowd.crowd):
    """
        这个类的作用是模拟在设定有密接的情况下，模拟传染病的过程
        次密接实在比较难刻画，不做了


        使用一个四维数组代表一个人，其中前两个维度代表这个人的 x 坐标和 y 坐标
        第三个维度代表这个人的状态

            其中 
            0 代表 s，即易感人群
            1 代表 i，即感染者
            2 代表 r，即康复
            3 代表密接，4代表次密接

        第四个维度代表这个人的感染天数（如果他感染）或是成为密接的天数
            s 的感染天数是 0
            r 的感染天数是 11（我们默认疾病10天内治愈）

            我们假定病毒的潜伏期是5天，即如果一个人的状态是 3(即密接)，那么他的天数这一栏里面的数字会是 0,1,2,3,4,5 . 这期间如果他感染了那么天数就清零，从感染日开始计算天数，反之也清零，状态设为 0(即易感人群)


        事实上我们只需要重构 Forward 和 Move 两个函数就可以
    """
    def __init__(self, size=4000, mod='dis', Emd = 5) -> None:
        super().__init__(size, mod)
        ## 下面这个参数是在指一个密接保持不感染的最大天数
        self.E_days = Emd

    def Forward(self, rate=0.5, dis=1.42):
        idx_S = np.argwhere(self.data[2,:] == 0).squeeze()
        idx_I = np.argwhere(self.data[2,:] == 1).squeeze()


        ## 现在让 I 去创造 密接
        if idx_I.size != 0:
            for i in np.nditer(idx_I):
                ## 这里 i 是这个感染者的编号
                pos_i = self.getLoc(i)
                for s in idx_S:
                    pos_s = self.getLoc(s)
                    if utils.distance(pos_i, pos_s) >= dis:
                        ## 如果这两个人足够远则无事发生
                        continue
                    else:
                        ## 否则这个可怜的人就会变成密接
                        self.data[2,s] = 3


        

        ## 下面原来的密接变成感染者
        idx_E = np.argwhere(self.data[2,:] == 3).squeeze()
        if idx_E.size == 0:
            return
        for e in np.nditer(idx_E):
            prob = rdm.rand(1)
            if prob > rate:
                ## 如果命好就无事发生，你还是密接
                continue
            else:
                ## 否则就变成感染者
                self.data[2,e] = 1

        

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
        if idx_I.size != 0:
            for i in np.nditer(idx_I):
                ## 感染者感染天数加 1
                self.data[3, i] += 1
                if self.data[3, i] == 11:
                    ## 出院！
                    self.data[2, i] = 2


        idx_E = np.argwhere(self.data[2,:] == 3).squeeze()
        if idx_E.size == 0:
            return
        for e in np.nditer(idx_E):
            if self.data[3, e] == self.E_days:
                ## 如果这哥们已经当了足够多天数的密接了，他就没有危险，仍然是易感人群
                self.data[3, e] = 0
                self.data[2, e] = 0
                continue

    def getData(self):
        idx_0 = np.argwhere(self.data[2,:] == 0).squeeze().size
        idx_1 = np.argwhere(self.data[2,:] == 1).squeeze().size
        idx_2 = np.argwhere(self.data[2,:] == 2).squeeze().size
        idx_3 = np.argwhere(self.data[2,:] == 3).squeeze().size
        return idx_0, idx_1, idx_2, idx_3