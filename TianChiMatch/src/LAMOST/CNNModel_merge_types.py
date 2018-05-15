# -*- coding: utf-8 -*-
'''
Created on 2018年3月6日

@author: zwp
'''

'''
单分类问题的例子
包括模型，训练，评价
模型持久化

使用卷积网络来实现

'''

import  tensorflow as tf;
import numpy as np;
from LAMOST import DataSource, DataSetProcess;
from tensorflow.python.platform import gfile;
import time;
########################## 参数    ##########################


def act_func(X):
    return tf.nn.relu(X);

# 特征数，输入张量的shape
feature_size = 2600;
# 标签数，输出张量的shape
label_size = 4;

# 输入向量调整，尺寸
cnn_input_size=int(np.ceil(np.sqrt(feature_size)));
# 输入向量调整，深度
cnn_input_deep=1;

# 卷积层1过滤器尺寸和深度
# 该过滤器步长为1，无填充
conv1_size=5;
conv1_deep=8;
conv1_step=1;

# 池化层1过滤器尺寸和深度
# 该过滤器步长2,0填充
pool1_size=2;
pool1_step=2;

# 卷积层2过滤器尺寸和深度
# 该过滤器步长为1，无填充
conv2_size=5;
conv2_deep=16;
conv2_step=1;

# 池化层2过滤器尺寸和深度
# 该过滤器步长2,无填充
pool2_size=2;
pool2_step=2


# 卷积层3过滤器尺寸和深度
# 该过滤器步长为2，无填充
conv3_size=2;
conv3_deep=32;
conv3_step=2;


# 全连接层隐层数与节点数
fc_hiddens = [128,16];




steps = 15000;
batch_size = 30;
learn_rate = 0.001;
learn_rate_decy = 0.96;

need_regular=True;
regular_lambda = 0.01;

need_val_avage=True;
move_avage_rate = 0.999;

model_save_path = 'value_cache/model_cnn.ckpt'

model_3o_graph_path = 'value_cache/graph_cnn_3o.pb'
model_uk_graph_path = 'value_cache/graph_cnn_uk.pb'

load_value = False;
need_train=True;
need_result_out=True;


########################## 输入转化 #######################

def change_to_cnn_input(x):
    '''
        将[None,feature_size]的输入转换到
        [None,cnn_input_size,cnn_input_size,cnn_input_deep]
        未满的部分用0填补
    '''
    x = np.array(x);
    batchs = x.shape[0];
    cx=np.zeros([batchs,cnn_input_size*cnn_input_size]);
    cx[:,0:feature_size]=x;
    cx = np.reshape(cx,
                    [-1,cnn_input_size,cnn_input_size,cnn_input_deep]);
    return cx;


def getGraphElement(gfile_path,ele_name,X):
    with gfile.FastGFile(gfile_path,'rb') as f:
        gd = tf.GraphDef();
        gd.ParseFromString(f.read());
    element = tf.import_graph_def(gd,input_map={'X':X},return_elements=[ele_name]);
    return element;

    
def evel(datasource):
    X = tf.placeholder(tf.float32, [None,cnn_input_size,cnn_input_size,cnn_input_deep], 'X');
    with tf.Session() as sess:
        py_uk=getGraphElement(model_uk_graph_path,'out_layer/add:0',X);
        py_uk = tf.nn.softmax(py_uk);
        py_3o = getGraphElement(model_3o_graph_path,'out_layer/add:0',X);
        py_3o = tf.nn.softmax(py_3o);
        data_size = datasource.data_size();
        xs,ys = datasource.getAllData();
        xs = change_to_cnn_input(xs);
        start = 0;
        cal_step = 1000;
        py=[];
        while start<data_size:
            end = min(start+cal_step,data_size);
            pyuk,py3o = sess.run([py_uk,py_3o],{X:xs[start:end]});
            pyuk=pyuk[0];
            py3o=py3o[0];
            for line in range(end-start):
                item=pyuk[line];
                if item[0]<=item[3]:
                    py.append(item);
                else:
                    py.append(py3o[line]);
            start+=cal_step;
    
    y = np.array(ys);
    py = np.array(py);
    
    mxpy=np.reshape(np.max(py,axis=1),(-1,1));
    py =(py/mxpy).astype(int);
    
    y_sum_types = np.sum(y,axis=0);
    py_sum_types= np.sum(py,axis=0);
    
    py_indexes = np.reshape(np.argmax(py, axis=1),(-1,1));
    y_indexes = np.reshape(np.argmax(y, axis=1),(-1,1));
    
    err=0;
    err_type=np.array([0,0,0,0],dtype=float);
    
    for i in range(data_size):
        yi = y_indexes[i];
        pyi = py_indexes[i];
        if yi != pyi:
            err+=1;
            err_type[yi]+=1;
    print('y=\t',y_sum_types);
    print('py=\t',py_sum_types);
    print('err=\t',err_type);
    tp = (y_sum_types-err_type);
    recall = np.divide(tp, y_sum_types, out=np.zeros_like(tp), where=y_sum_types!=0);
    prec = np.divide(tp, py_sum_types, out=np.zeros_like(tp), where=py_sum_types!=0);
    
    print('recall\t',recall);
    print('prec\t',prec);
    tmp1 = 2*recall*prec;
    tmp2 = recall+prec;
    macro_f1 = np.mean(np.divide(tmp1,tmp2,out=np.zeros_like(tmp1),where=tmp2!=0));
    print('all=%d true=%d err=%d pr=%.2f%%,macro_f1=%.3f'%(data_size,data_size-err,err,err*100.0/data_size,macro_f1));
    return py;    

########################## 计算部分 #######################

def calculate(datasource):
    X = tf.placeholder(tf.float32, [None,cnn_input_size,cnn_input_size,cnn_input_deep], 'X');
    with tf.Session() as sess:
        py_uk =getGraphElement(model_uk_graph_path,'out_layer/add:0',X);
        py_uk = tf.nn.softmax(py_uk);
        py_3o = getGraphElement(model_3o_graph_path,'out_layer/add:0',X);
        py_3o = tf.nn.softmax(py_3o);
        
        xs = datasource.getAllDataX();
        data_size= datasource.data_size();
        start =0;
        cal_step=100;
        xs = change_to_cnn_input(xs);
        py=[];
        while start<data_size:
            end = min(start+cal_step,data_size);
            pyuk,py3o = sess.run((py_uk,py_3o),{X:xs[start:end]});
            pyuk=pyuk[0];
            py3o=py3o[0];
            for line in range(end-start):
                item=pyuk[line];
                if item[0]<=item[3]:
                    py.append(item);
                else:
                    py.append(py3o[line]);
            start+=cal_step;
    return py;




base_path=r'/home/zwp/work/Dataset/tianci/LAMOST';
train_data_index = base_path+r'/index_train.csv';
train_data_zip = base_path+r'/first_train_data_20180131.zip';

test_data_index = base_path+r'/index_test.csv';
test_data_zip = base_path+r'/first_train_data_20180131.zip';


result_test_index=base_path+r'/first_test_index_20180131.csv';
result_test_data_zip=base_path+r'/first_test_data_20180131.zip';
result_test_out_path=base_path+r'/test_result.csv';


def run():

    print('\n加载训练数据')
    train_ds = DataSource.DataSource(train_data_index,train_data_zip);

    print('\n开始测试')
    train_ds.reload_index(test_data_index);
    print('开始测评')
    evel(train_ds);
    
    if need_result_out:
        print('\n开始加载比赛数据集')
        result_ds=DataSource.DataSource(result_test_index,result_test_data_zip);
        indexids=result_ds.getAllIndexID();
        print('计算比赛数据')
        py=calculate(result_ds);
        print('创建比赛文件')
        DataSetProcess.create_result_csv(indexids, py, result_test_out_path);
    
    
    pass;



if __name__ == '__main__':
    run();
    pass