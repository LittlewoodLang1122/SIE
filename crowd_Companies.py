##version2.0
import numpy as np
from numpy import random as rdm
from matplotlib import pyplot as plt
import crowd
import utils


class crowd_C(crowd.crowd):
    """
        这个类是 在基础模型的基础上，设计四个可以视作公司的地方。每隔一天所有人都会聚集在四个地点附近，模拟上班和下班的过程
        使用一个五维数组代表一个人，其中前两个维度代表这个人的 x 坐标和 y 坐标
        第三个维度代表这个人的状态

            其中 
            0 代表 s，即易感人群
            1 代表 i，即感染者
            2 代表 r，即康复
            3 代表密接，4代表次密接
        第四个维度代表这个人的感染天数（如果他感染）
            s 的感染天数是 0
            r 的感染天数是 11（我们默认疾病10天内治愈）
        第五个维度代表这个人的从属的公司序号i（i =1,2,3,4）


    """

    def __init__(self, size=4000, mod='dis') -> None:
        super().__init__(size, mod)
        extra_data = rdm.randint(1,5,size = [1,self.size])
        self.data = np.r_[self.data, extra_data]
        self.home_loc = self.data[0:2,:].copy()

    def Move(self,day):
        """
        设第一次迭代是day1，则奇数天去公司，偶数天回去在家附近
        
        """
        if day % 2 == 0:
             ##去公司的情况
            
            idx_c1 = np.argwhere(self.data[4,:] == 1).squeeze()
            idx_c2 = np.argwhere(self.data[4,:] == 2).squeeze()
            idx_c3 = np.argwhere(self.data[4,:] == 3).squeeze()
            idx_c4 = np.argwhere(self.data[4,:] == 4).squeeze()

            self.data[0,idx_c1] = utils.normal_lu(idx_c1.size, lower=0, upper=500, prop=4) 
            self.data[1,idx_c1] = utils.normal_lu(idx_c1.size, lower=0, upper=500, prop=4) 
            self.data[0,idx_c2] = utils.normal_lu(idx_c2.size, lower=0, upper=500, prop=4) 
            self.data[1,idx_c2] = utils.normal_lu(idx_c2.size, lower=-500, upper=0, prop=4) 
            self.data[0,idx_c3] = utils.normal_lu(idx_c3.size, lower=-500, upper=0, prop=4) 
            self.data[1,idx_c3] = utils.normal_lu(idx_c3.size, lower=-500, upper=0, prop=4) 
            self.data[0,idx_c4] = utils.normal_lu(idx_c4.size, lower=-500, upper=0, prop=4) 
            self.data[1,idx_c4] = utils.normal_lu(idx_c4.size, lower=0, upper=500, prop=4) 
        else :
            ##回家的情况
            for i in range(2):
                self.data[i,:] = self.home_loc[i,:] + utils.normal_lu(self.size, lower=10, upper=10, prop=0.5)
               
        idx_I = np.argwhere(self.data[2,:] == 1).squeeze()
        for i in idx_I:
            ## 感染者感染天数加 1
            self.data[3, i] += 1
            if self.data[3, i] == 14:
                ## 出院！
                self.data[2, i] = 2 

            recover_or_not = rdm.binomial(n = 1, p = (self.data[3, i])**2 / 196 , size = 1)
            if recover_or_not == 1:
                self.data[2, i] = 2 
                 
        
        







