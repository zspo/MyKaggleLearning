# -*- coding: utf-8 -*-
'''
Created on 2018年2月28日

@author: zwp
'''


import tensorflow as tf;
from LAMOST import DataSource, DataSetProcess
import time;
import numpy as np;
class BPModel():
    '''
        构造一个双隐层模型，最后softmax输出
    '''
    X = None; # Input feature
    Y = None; # Input label
    loss = None; # Loss Node
    def __init__(self,f_size,l_size,hs_size,act_func=None,rou=1.0):
        '''
        f_size:特征数
        l_size:标签数
        hs_s:双隐层单元数
        '''
        self.X = tf.placeholder(tf.float32,[None,f_size],name='X');
        self.Y = tf.placeholder(tf.float32,[None,l_size],'Y');
        self.loss = self.create_model(f_size, l_size, hs_size, act_func, rou);
        pass;
    

    def create_model(self,f_size,l_size,hs_size,act_func=None,rou=1.0):
        if act_func == None:
            act_func = tf.tanh;
        self.global_step = tf.Variable(0);
        #  f_size * hs_size_0
        W1 = tf.Variable(tf.random_normal([f_size,hs_size[0]],stddev=rou, 
                                          dtype=tf.float32),name='W1');
        b1 = tf.Variable(tf.random_normal([hs_size[0]],stddev=rou, 
                                          dtype=tf.float32),name='b1');
        
        #  hs_size_0 * hs_size_1                        
        W2 = tf.Variable(tf.random_normal([hs_size[0],hs_size[1]],stddev=rou, 
                                          dtype=tf.float32),name='W2');
        b2 = tf.Variable(tf.random_normal([hs_size[1]],stddev=rou, 
                                          dtype=tf.float32),name='b2');
        #  hs_size_1 * l_size                                                            
        W3 = tf.Variable(tf.random_normal([hs_size[1],l_size],stddev=rou, 
                                          dtype=tf.float32),name='W3');
        b3 = tf.Variable(tf.random_normal([l_size],stddev=rou, 
                                          dtype=tf.float32),name='b3');        
        
        h1 = act_func(tf.matmul(self.X,W1)+b1);
        h2 = act_func(tf.matmul(h1,W2)+b2);
        out_line_real = tf.matmul(h2,W3)+b3 # 没有激活函数，用于softmax
        
        h1drop = tf.nn.dropout(h1,0.2);
        h2_d1 = act_func(tf.matmul(h1drop,W2)+b2);
        h2d_d1 = tf.nn.dropout(h2_d1,0.2);
        out_line_train = tf.matmul(h2d_d1,W3)+b3 # 没有激活函数，用于softmax
        py_=tf.nn.softmax(out_line_train);
#       train_loss = -tf.reduce_mean(self.Y * tf.log(py_)+(1-self.Y) * tf.log((1-py_)));
        train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=out_line_real, name='loss'));
        reg = tf.contrib.layers.l2_regularizer(.01);
        train_loss = train_loss + reg(W1);
        train_loss = train_loss + reg(W2);
#         train_loss = train_loss + tf.contrib.layers.l2_regularizer(.01)(W3);
        self.py = tf.nn.softmax(out_line_real);
        return train_loss;
    
    def train(self,datasource,learn_rate,steps,batch_size,load_value=False):
        
        data_size = datasource.data_size();
        
        lr = tf.train.exponential_decay(learn_rate, self.global_step, 100, 0.96, staircase=True);
        train_step = tf.train.AdamOptimizer(lr).minimize(self.loss,global_step=self.global_step);
        saver = tf.train.Saver(tf.global_variables());
        with tf.Session() as sess:
            if load_value:
                saver.restore(sess, 'value_cache/bpmode1.ckpt');
            else:
                sess.run(tf.global_variables_initializer());
            now = time.time();
            for i in range(steps):
                start = (i*batch_size) % data_size;
                end = min(start+batch_size,data_size);
                xs,ys = datasource.getDataXY(start,end);
                _,loss,py,y=sess.run((train_step,self.loss,self.py,self.Y),{self.X:xs,self.Y:ys});
                if i%20 ==0:
                    print('step=%d loss=%.5f time=%.2f'%(i,loss,time.time()-now));
                    # print(sess.run(global_step));
                    # print(py,y);
                    now = time.time();
            saver.save(sess,'value_cache/bpmode1.ckpt');
        print('finish!! ');

    def calculate(self,datasource,start=0,end=0):
        saver = tf.train.Saver(tf.global_variables());
        with tf.Session() as sess:
            saver.restore(sess, 'value_cache/bpmode1.ckpt');
            if start==0 and end ==0:
                xs,ys = datasource.getAllData();
            else:
                xs,ys = datasource.getDataXY(start,end);
            loss,py,y= sess.run((self.loss,self.py,self.Y),{self.X:xs,self.Y:ys});
            print('calculate result== %.5f'%(loss));
            print(py,y);
        return (loss,py,y);
    
    def cal_py_out(self,datasource,start=0,end=0):
        saver = tf.train.Saver(tf.global_variables());
        with tf.Session() as sess:
            saver.restore(sess, 'value_cache/bpmode1.ckpt');
            if start==0 and end ==0:
                xs = datasource.getAllDataX();
            else:
                xs = datasource.getData_X(start,end);
            py= sess.run((self.py),{self.X:xs});
            print('cal_py_out ',py);
        return py;
###################################### end class ##################################


feature_size = 2600;
label_size = 4;
hidden_size = (128,32);

learn_rate = 0.0007;
steps = 3000;
batch_size = 30;

load_values=False;
need_train=True;
need_result_out=False
rou = 0.05;
def act_func(X):
    return tf.tanh(X);


def evel(py,y):
    py = np.array(py);
    mxpy=np.reshape(np.max(py,axis=1),(-1,1));
    py =(py/mxpy).astype(int);
    y = np.array(y);
    y_sum_types = np.sum(y,axis=0);
    py_sum_types= np.sum(py,axis=0);
    delta=y-py;
    max_ids=np.argmax(delta, axis=1);
    all=max_ids.shape[0];
    err=0;
    err_type=np.array([0,0,0,0],dtype=float);
    for item in range(all):
        if not delta[item,max_ids[item]] == 0:
            err+=1;
            err_type[max_ids[item]]+=1.0;
    print(y_sum_types,py_sum_types);
    tp = (y_sum_types-err_type);
    recall = np.divide(tp, y_sum_types, out=np.zeros_like(tp), where=y_sum_types!=0);
    prec = np.divide(tp, py_sum_types, out=np.zeros_like(tp), where=py_sum_types!=0);
    
    print(recall,prec);
    tmp1 = 2*recall*prec;
    tmp2 = recall+prec;
    macro_f1 = np.mean(np.divide(tmp1,tmp2,out=np.zeros_like(tmp1),where=tmp2!=0));
    print(py);
    print('all=%d true=%d err=%d pr=%.2f%%,macro_f1=%.3f'%(all,all-err,err,err*100.0/all,macro_f1));
    print('err_type=',err_type);
    return py;




base_path=r'/home/zwp/work/Dataset/tianci/LAMOST';
train_data_index = base_path+r'/index_train.csv';
train_data_zip = base_path+r'/first_train_data_20180131.zip';

test_data_index = base_path+r'/index_test.csv';
test_data_zip = base_path+r'/first_train_data_20180131.zip';


result_test_index=base_path+r'/first_test_index_20180131.csv';
result_test_data_zip=base_path+r'/first_test_data_20180131.zip';
result_test_out_path=base_path+r'/test_result.csv';

ttt = base_path+r'/tt.csv';
def run():
    
    print('\n加载训练数据')
    train_ds = DataSource.DataSource(train_data_index,train_data_zip);
    model = BPModel(feature_size,label_size,hidden_size,act_func,rou);
    if need_train:
        print('\n开始训练')
        model.train(train_ds, learn_rate, steps, batch_size,load_values);
    print('\n开始测试')
    train_ds.reload_index(test_data_index);
    print('开始测试输出')
    _,py,y=model.calculate(train_ds);
    print('开始评价')
    evel(py,y);
    
    if need_result_out:
        print('\n开始加载比赛数据集')
        result_ds=DataSource.DataSource(result_test_index,result_test_data_zip);
        indexids=result_ds.getAllIndexID();
        print('计算比赛数据')
        py=model.cal_py_out(result_ds);
        print('创建比赛文件')
        DataSetProcess.create_result_csv(indexids, py, result_test_out_path);

    pass;


if __name__ == '__main__':
    run();
    pass