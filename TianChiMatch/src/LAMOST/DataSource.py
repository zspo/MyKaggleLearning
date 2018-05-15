# -*- coding: utf-8 -*-
'''
Created on 2018年2月28日

@author: zwp
'''
import zipfile;
import pandas as pd;
import numpy as np;


class DataSource(object):
    f_index = None;
    f_data_zip =None;
    size=0;
    
    '''
        path_index:a path ofcsv file
        path_dzip: a path of zip file
    '''
    def __init__(self,path_index,path_dzip):
        self.f_index = pd.read_csv(path_index);
        self.f_data_zip = zipfile.ZipFile(path_dzip,'r');
        self.size = len(self.f_index);
    
    def getLabel(self,dtype):
        tmp=[0,0,0,0];
        tmp[dtype]=1;
        return tmp;
    
    def data_preprocess(self,x):
        return (x-np.mean(x))/np.std(x);
#         x = np.array(x);
#         amax=np.max(x);
#         amin=np.min(x);
#         if amax>0:
#             np.divide(x,amax,out=x,where=x>0);
#         if amin<0:
#             np.divide(x,-amin,out=x,where=x<0);
#         return x;
    def getFeature(self,did):
        dstr = self.f_data_zip.read('%s.txt'%(did)).decode('utf-8');
        ori = np.fromstring(dstr,sep=',');
        aft = self.data_preprocess(ori);
        return aft;
    
    def getDataXY(self,start,end):
        indexs = self.f_index;
        if start<0 or end < 0 or end > self.size:
            raise ValueError('start=%d end=%d error'%(start,end));
        features = [];
        labels = [];
        for item in indexs.values[start:end]:
            features.append(self.getFeature(item[0]));
            labels.append(self.getLabel(item[1]));
        return np.array(features),np.array(labels);

    def getDataX(self,start,end):
        indexs = self.f_index;
        if start<0 or end < 0 or end > self.size:
            raise ValueError('start=%d end=%d error'%(start,end));
        features = [];
        for item in indexs.values[start:end]:
            features.append(self.getFeature(item[0]));
        return np.array(features);

    def getIndexID(self,start,end):
        indexs = self.f_index;
        if start<0 or end < 0 or end > self.size:
            raise ValueError('start=%d end=%d error'%(start,end));
        ids = [];
        for item in indexs.values[start:end]:
            ids.append(item[0]);
        return ids;
    
    def getAllData(self):
        return self.getDataXY(0, self.size);
    def getAllDataX(self):
        return self.getDataX(0, self.size);
    def getAllIndexID(self):
        return self.getIndexID(0, self.size);
    
    def data_size(self):
        return self.size;
    
    def reload_index(self,path_index):
        self.f_index = pd.read_csv(path_index);
        self.size = len(self.f_index);
        
    def reload_dzip(self,path_dzip):
        if not self.f_data_zip is None:
            self.f_data_zip.close();
        self.f_data_zip = zipfile.ZipFile(path_dzip,'r');
    

        
            
    def close(self):
#         if not self.f_index is None:
#             self.f_index.close();
        if not self.f_data_zip is None:
            self.f_data_zip.close();
    
            
    pass;



