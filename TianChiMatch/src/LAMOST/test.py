# -*- coding: utf-8 -*-
'''
Created on 2018年2月28日

@author: zwp
'''
from LAMOST import DataSource;
import time ;
zip_path = r'/home/zwp/work/Dataset/tianci/first_train_data_20180131.zip';
csv_path = r'/home/zwp/work/Dataset/tianci/first_train_index_20180131.csv';
csv_path_type = r'/home/zwp/work/Dataset/tianci/first_train_index_types.csv';
if __name__ == '__main__':

    datasource = DataSource.DataSource(csv_path_type,zip_path);
    
    now = time.time();
    print(datasource.getDataXY(1, 200));
    print("%.2f"%(time.time()-now));
    datasource.close();
    pass