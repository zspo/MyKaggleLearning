# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import xgboost as xgb
import time
import random
import numpy as np
import math
from sklearn.linear_model import LinearRegression

 

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。

def	timestamp_datetime(value):
	format = '%Y-%m-%d %H:%M:%S'
	value = time.localtime(value)
	dt = time.strftime(format, value)
	return dt
	
def time_S(data): #时间日期处理
	data['time']=data.TIME.apply(lambda x:timestamp_datetime(x))
	data['month']=data.time.apply(lambda x:int(x[5:7]))
	data['day']=data.time.apply(lambda x:int(x[8:10]))
	data['hour']=data.time.apply(lambda x:int(x[11:13]))
	data['minute']=data.time.apply(lambda x:int(x[14:16]))
	return data

def jingweigroupcha(data):
	data['ss']=data.index
	asd=data.groupby(by=(['TERMINALNO','TRIP_ID']),as_index=False)['LONGITUDE'].diff()
	asd=asd.rename('jingducha')
	asd=pd.DataFrame(asd)
	asd['ss']=range(data['ss'].max()+1)
	asd=asd.fillna(0)
	data=pd.merge(data,asd,on=['ss'])
	asd=data.groupby(by=(['TERMINALNO','TRIP_ID']),as_index=False)['LATITUDE'].diff()
	asd=asd.rename('weiducha')
	asd=pd.DataFrame(asd)
	asd['ss']=range(data['ss'].max()+1)
	asd=asd.fillna(0)
	data=pd.merge(data,asd,on=['ss'])
	del asd ,data['ss']
	return data
 	
'''ID唯一训练，测试集
TERMINALNO       int64
TIME             int64
TRIP_ID          int64
LONGITUDE      float64
LATITUDE       float64
DIRECTION        int64
SPEED          float64
CALLSTATE        int64
Y              float64
'''
data = pd.read_csv(path_train)
df = pd.DataFrame()
df=data[['TERMINALNO','TRIP_ID']].drop_duplicates(subset=['TERMINALNO','TRIP_ID'])
df['Id']=df.TERMINALNO



def train_y(df): #训练目标
	A=data.groupby(by=(['TERMINALNO']))['Y'].max()
	A=pd.DataFrame(A)
	A['Id']=A.index
	df=pd.merge(df,A,on=['Id'],how='left')
	df['train_y']=df.Y
	df=df.drop(['Y'],axis=1)
	
	return df

def DATEtime(df):
	A=time_S(data)
	B=A[['TERMINALNO','TRIP_ID','day','hour','minute']]
	df=pd.merge(df,B,on=['TERMINALNO','TRIP_ID'])
	del A,B
	return df
	


def speed_max(df):#最大速度
	A=data.groupby(by=(['TERMINALNO']))['SPEED'].max()
	A=pd.DataFrame(A)
	A['Id']=A.index
	df=pd.merge(df,A,on=['Id'],how='left')
	df['speed_max']=df.SPEED
	df=df.drop(['SPEED'],axis=1)
	
	return df

	
def speed_min(df):#最小速度
	A=data.groupby(by=(['TERMINALNO']))['SPEED'].min()
	A=pd.DataFrame(A)
	A['Id']=A.index
	df=pd.merge(df,A,on=['Id'],how='left')
	df['speed_min']=df.SPEED
	df=df.drop(['SPEED'],axis=1)
	
	return df

def speed_mean(df):#平均速度
	A=data.groupby(by=(['TERMINALNO']))['SPEED'].mean()
	A=pd.DataFrame(A)
	A['Id']=A.index
	df=pd.merge(df,A,on=['Id'],how='left')
	df['speed_mean']=df.SPEED
	df=df.drop(['SPEED'],axis=1)
	
	return df
	
def height_max(df):#最大高度
	A=data.groupby(by=(['TERMINALNO']))['HEIGHT'].max()
	A=pd.DataFrame(A)
	A['Id']=A.index
	df=pd.merge(df,A,on=['Id'],how='left')
	df['height_max']=df.HEIGHT
	df=df.drop(['HEIGHT'],axis=1)
	
	return df

def height_min(df):#最小高度
	A=data.groupby(by=(['TERMINALNO']))['HEIGHT'].min()
	A=pd.DataFrame(A)
	A['Id']=A.index
	df=pd.merge(df,A,on=['Id'],how='left')
	df['height_min']=df.HEIGHT
	df=df.drop(['HEIGHT'],axis=1)
	
	return df
	
def height_mean(df):#平均高度
	A=data.groupby(by=(['TERMINALNO']))['HEIGHT'].mean()
	A=pd.DataFrame(A)
	A['Id']=A.index
	df=pd.merge(df,A,on=['Id'],how='left')
	df['height_mean']=df.HEIGHT
	df=df.drop(['HEIGHT'],axis=1)
	
	return df

def call_count(df):#电话分类统计
	for i in ([0,1,2,3,4]):
		dfcall=data[(data['CALLSTATE']==i)]
		callcount=dfcall.groupby(by=('TERMINALNO'))['CALLSTATE'].count()
		callcount=pd.DataFrame(callcount)
		callcount['Id']=callcount.index
		df=pd.merge(df,callcount,on=['Id'],how='left')
		df['CALLSTATE'+str(i)]=df.CALLSTATE
		del df['CALLSTATE']
		del callcount
	return df
	
def notnull(df):#空值转为0
	df=df.fillna(0)
	
	return df
	

 	

def height_sum(df):#高度变化绝对值之和
	global data
	data['asd']=data.HEIGHT.diff()
	data['asd']=data.asd.apply(lambda x:np.abs(x))
	data=data.fillna(1)
	A=data.groupby(by=(['TERMINALNO']))['asd'].sum()
	A=pd.DataFrame(A)
	A['Id']=A.index
	A=A.fillna(1)
	
	df=pd.merge(df,A,on=['Id'],how='left')
	df['height_sum']=df.asd
	df=df.drop(['asd'],axis=1)
	
	return df
def speed_sum(df):#速度变化绝对值之和
	global data
	data['asd']=data.SPEED.diff()
	data['asd']=data.asd.apply(lambda x:np.abs(x))
	data=data.fillna(1)
	A=data.groupby(by=(['TERMINALNO']))['asd'].sum()
	A=pd.DataFrame(A)
	A['Id']=A.index
	A=A.fillna(1)
	
	df=pd.merge(df,A,on=['Id'],how='left')
	df['speed_sum']=df.asd
	df=df.drop(['asd'],axis=1)
	
	return df

def direction_sum(df):#角度变化绝对值之和
	global data
	data['asd']=data.DIRECTION.diff()
	data['asd']=data.asd.apply(lambda x:np.abs(x))
	data=data.fillna(1)
	A=data.groupby(by=(['TERMINALNO']))['asd'].sum()
	A=pd.DataFrame(A)
	A['Id']=A.index
	A=A.fillna(1)
	
	df=pd.merge(df,A,on=['Id'],how='left')
	df['direction_sum']=df.asd
	df=df.drop(['asd'],axis=1)
	
	return df	


	
def log_c(df):#log转化
	
	for i,j in zip(['direction_sum','speed_sum','height_sum','speed_max','speed_mean', 'height_max',  'height_mean',],['direction_sum_log','speed_sum_log','height_sum_log','speed_max_log','speed_mean_log',' height_max_log',' height_mean_log',]):
		df[j]=df[i].apply(lambda x:math.log(np.abs(x)+1))
	return df
		
def smean_hmean(df):#平均速度*平均高度
	df['smean_hmean']=df.speed_mean*df.height_mean	
	return df
	
	
def start_up(df):#车辆启停次数
	asd=data[(data.SPEED)==0].groupby(by=(['TERMINALNO']))['SPEED'].count()
	asd=pd.DataFrame(asd)
	asd['Id']=asd.index
	df=pd.merge(df,asd,on=['Id'],how='left')
	df['start_up']=df.SPEED
	df=df.drop(['SPEED'],axis=1)
	df=df.fillna(0)
	del asd
	return df
	


	#采样数
def luchengFZjisuan1(df):#每段路程角度，高度，速度均值
	for i in ['DIRECTION','HEIGHT','SPEED']:
		asd=data.groupby(by=(['TERMINALNO','TRIP_ID']),as_index=False)[i].agg({str(i)+'TP':'mean'})
		df = pd.merge(df, asd, on=['TERMINALNO','TRIP_ID'], how='left')
		del asd
	return df
def luchengJWFZjisuan1(df): #每段路程的经纬度均值
	for i in ['LONGITUDEc','LATITUDEc']:
		asd=data.groupby(by=(['TERMINALNO','TRIP_ID']),as_index=False)[i].agg({str(i)+'JW':'mean'})
		df = pd.merge(df, asd, on=['TERMINALNO','TRIP_ID'], how='left')
		del asd
	return df
	
def luchengJWFZjisuan2(df): #每段路程的经纬度差值均值
	for i in ['jingducha','weiducha']:
		asd=data.groupby(by=(['TERMINALNO','TRIP_ID']),as_index=False)[i].agg({str(i)+'JWC':'mean'})
		df = pd.merge(df, asd, on=['TERMINALNO','TRIP_ID'], how='left')
		del asd
	return df	

def luchengFZjisuan2(df):#每段路程的平均角度,速度*路程平均高度
	for i in ['DIRECTIONTP','SPEEDTP']:
		df[str(i)+'_xs']=df[i]*df.HEIGHTTP
	return df	

def luchengFZjisuan3(df):#每段路程的平均角度,速度*平均高度
	for i in ['DIRECTIONTP','SPEEDTP']:
		for j in ['height_max','height_min','height_mean']:
			df[str(i)+str(j)+'_xs']=df[i]*df.HEIGHTTP
	return df
	
	
def call_H_xc(df): #电话状态数与各值相乘，以去除'HEIGHTTP','DIRECTIONTP','SPEEDTP',
	for i in ['CALLSTATE0','CALLSTATE1','CALLSTATE2','CALLSTATE3','CALLSTATE4']:
		for j in ['height_max','height_min','height_mean']:
			df[str(i)+str(j)+'_xs']=df[i]*df[j]
	
	return df
	

	

#驾龄，加速度，天，时，秒，高度*速度，转弯速度,每个用户通话类等于3的count,每个用户的Hight总和
def	read_csv():
	global df
	global df2
	global data
	data = jingweigroupcha(data)
	df=train_y(df)
	df=speed_max(df)
	df=speed_min(df)
	df=speed_mean(df)
	df=height_max(df)
	df=height_min(df)
	df=height_mean(df)
	df=call_count(df)
	df=height_sum(df)
	df=speed_sum(df)
	df=direction_sum(df)
	df=smean_hmean(df)
	df=start_up(df)
	#df=luchengFZjisuan1(df)
	#df=luchengFZjisuan2(df)
	#df=luchengFZjisuan3(df)
	#df=call_H_xc(df)
	#df=pd.merge(df,data,on=['TERMINALNO','TRIP_ID'], how='left')
	#数据归一化
	df = (df - df.min()) / (df.max() - df.min())
	
	df=notnull(df)
	Y_mean=data.Y.mean()
	print('train Y.mean')
	print(data.Y.mean())
	print(data.Y.max())
	print(data.Y.min())
	del data  #清空data内存
	X=df.drop('train_y',axis=1)
	X=X.drop('TERMINALNO',axis=1)
	X=X.drop('TRIP_ID',axis=1)
	#X=X.drop('Id',axis=1)
	Y=df.train_y
	#Y=Y.apply(lambda x:0 if x==0 else 1) #分类模型 用概率排序
	print(X.info())
	
	data = pd.read_csv(path_test) #test内存
	df2 = pd.DataFrame()
	df2=data[['TERMINALNO','TRIP_ID']].drop_duplicates(subset=['TERMINALNO','TRIP_ID'])
	df2['Id']=df2.TERMINALNO
	df2=speed_max(df2)
	df2=speed_min(df2)
	df2=speed_mean(df2)
	df2=height_max(df2)
	df2=height_min(df2)
	df2=height_mean(df2)
	df2=call_count(df2)
	df2=height_sum(df2)
	df2=speed_sum(df2)
	df2=direction_sum(df2)
	df2=smean_hmean(df2)
	df2=start_up(df2)
	#df2=luchengFZjisuan1(df2)
	#df2=luchengFZjisuan2(df2)
	#df2=luchengFZjisuan3(df2)
	#df2=call_H_xc(df2)
	#df2=pd.merge(df2,data,on=['TERMINALNO','TRIP_ID'], how='left')
	#数据归一化
	df2 = (df2 - df2.min()) / (df2.max() - df2.min())
	df2=df2.drop('TERMINALNO',axis=1)
	df2=df2.drop('TRIP_ID',axis=1)
	df3=df2 #方便提交
	#df2=df2.drop('Id',axis=1)
	df2=notnull(df2)
	del data
	'''
	feature=['TERMINALNO','TIME', 'LONGITUDE', 'LATITUDE', 'DIRECTION','HEIGHT', 'SPEED', 'CALLSTATE']#'hour','H-S']#'CALLSTATE0','CALLSTATE1','CALLSTATE2','CALLSTATE3','CALLSTATE4',]'TERMINALNO', 'TRIP_ID', 
	target = 'Y'
	
	

    
	
	X = data[feature]
	Y = data[target]
	'''
	
	model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.2, max_delta_step=0, max_depth=7, min_child_weight=1, missing=None, n_estimators=50, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1) #xgb回归
	
	#model=xgb.XGBClassifier()#xgboost分类
	#model =LinearRegression() #线性回归
	model.fit(X,Y)
	
	
	
	
	
	
	pred=model.predict(df2)
	#pred=model.predict_proba(df2)[:, 1]
	
	
	
	
	
	df3['Pred']=pd.Series(pred)
	'''
	test['Id']=test['TERMINALNO']
	df=test[['Id','Pred']]
	df2=df.groupby(by=('Id'))['Pred'].min()
	df=pd.DataFrame()
	df['Pred']=df2
	#df.to_csv("model/first.csv")
	#df=pd.read_csv("model/first.csv")
	'''
	#df2['Pred']=df2.Pred.apply(lambda x:0 if x<0.005 else x)
	df3=df3[['Id','Pred']]
	df3=df3.drop_duplicates(subset='Id')
	#df3['Pred']=df3.Pred.apply(lambda x:(np.abs(x+Y_mean) if x >0.4 else np.abs(-x+Y_mean))
	df3.to_csv("model/first.csv",index=False)
	#print (model.get_params)
	predictors = [i for i in X.columns]
	feat_imp = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
	print(feat_imp)
	#print (model.feature_importances_())
	
	#print (df)
	
	



if __name__ == "__main__":
    print("****************** start min **********************")
    # 程序入口
    read_csv()
