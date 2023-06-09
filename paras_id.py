import numpy as np
import scipy as sp
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import math
from lmfit import minimize, Parameters        # lmfit的最小二乘可以调节bounds，scipy中的不可调
import lmfit

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


class Para_id:
    def __init__(self, Uoc = 3, I = 1, R0 = 2) -> None:
        self.Uoc = Uoc
        self.I = I
        self.R0 = R0
    
    def func1(self,t,R1,R2,tao1, tao2):
        return self.Uoc-self.I*self.R0-self.I*R1*(1-math.e**(-t/tao1))-self.I*R2*(1-math.e**(-t/tao2))
    
    def func2(self, t,R1,R2,tao1, tao2):
        return self.Uoc-self.I*R1*math.e**(-t/tao1)-self.I*R2*math.e**(-t/tao2)
    
    def create_params(self):
        params = lmfit.create_params(R1={'value': 0.001, 'min': 0},
                       R2={'value': 0.001, 'min': 0},
                       tao1={'value': 0.1, 'min': 0},
                       tao2={'value': 0.1, 'min':0})
        return params
    
    def resid(self, params, t, V1, V2):
        R1 = params['R1']
        R2 = params['R2']
        tao1 = params['tao1']
        tao2 = params['tao2']
        diff1 = self.func1(t,R1,R2,tao1, tao2) - V1[t]
        diff2 = self.func2(t,R1,R2,tao1, tao2) - V2[t]
        return np.concatenate((diff1, diff2))

    def fit(self, params, t, V1, V2):
        out = minimize(self.resid, params, args=(t, V1, V2))
        return out
    
    def fit_plot(self, out, X_data, V1, V2):
        plt.figure(figsize = (14,5))
        plt.subplot(121)
        plt.scatter(X_data, V1)
        plt.plot(np.array(X_data), self.func1(np.array(X_data),
                                         out.params['R1'].value,
                                         out.params['R2'].value,
                                         out.params['tao1'].value,
                                         out.params['tao2'].value),
                 color='red', label="Fitted curve")

        # plt.figure()
        plt.subplot(122)
        plt.scatter(X_data, V2)
        plt.plot(np.array(X_data), self.func2(np.array(X_data),
                                         out.params['R1'].value,
                                         out.params['R2'].value,
                                         out.params['tao1'].value, 
                                         out.params['tao2'].value),
                 color='red', label="Fitted curve")