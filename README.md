# kaggle_restaurant  
如何进入kaggle10%，写的很好：  http://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/  
git clone https://github.com/wizardforcel/data-science-notebook #如何进入kaggle10%    
MXNET的对应房价的实现及论坛讨论，学习如何调参： http://zh.gluon.ai/chapter_supervised-learning/kaggle-gluon-kfold.html 参考discussion思路      
xgboost https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.9876270.0.0.38360a65Fcno7B&raceId=&postsId=2572      
GBDT调参: https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/   
XGBoost调参：https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ （调了半天没有效果一般，不知道为啥，，，）    
LSTM代码参考：

GBDT调参结果：
import glob, re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import time

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('./train_featured.csv')
col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors']]
target='visitors'
predictors = col
def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5
