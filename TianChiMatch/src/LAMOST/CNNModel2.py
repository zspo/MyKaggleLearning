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
from LAMOST import DataSource, DataSetProcess
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
conv1_size=3;
conv1_deep=8;
conv1_step=1;

# 池化层1过滤器尺寸和深度
# 该过滤器步长2,0填充
pool1_size=2;
pool1_step=2;

# 卷积层2过滤器尺寸和深度
# 该过滤器步长为1，无填充
conv2_size=3;
conv2_deep=16;
conv2_step=1;

# 池化层2过滤器尺寸和深度
# 该过滤器步长2,0填充
pool2_size=2;
pool2_step=2


# 卷积层3过滤器尺寸和深度
# 该过滤器步长为2，无填充
conv3_size=3;
conv3_deep=32;
conv3_step=3;


# 全连接层隐层数与节点数
fc_hiddens = [256,32];




steps = 15000;
batch_size = 30;
learn_rate = 0.001;
learn_rate_decy = 0.96;

need_regular=True;
regular_lambda = 0.01;

need_val_avage=True;
move_avage_rate = 0.999;

model_save_path = 'value_cache/model_cnn2.ckpt'

load_value = False;
need_train=True;
need_result_out=False;
########################## 模型部分 #######################

def get_weight_variable(shape,regularizer=None):
    '''
    获取权重函数，如果regularizer不是None
    则在损失函数中添加正则化
    '''
    weights = tf.get_variable('weights',
                                shape,
                                initializer=tf.truncated_normal_initializer( stddev=0.05));
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights));
    return weights;

# 定义模型函数，
def get_inference(X,act_func,regularizer):
    '''
    定义模型函数
    '''
    '''    

    '''
    with tf.variable_scope('layer1-conv1',reuse=tf.AUTO_REUSE):
        con_shape=[conv1_size,conv1_size,cnn_input_deep,conv1_deep];
        con_weights=get_weight_variable(con_shape, None);
        conv_biases=tf.get_variable('biases',[conv1_deep],
                                     initializer=tf.constant_initializer(0.0));
        conv = tf.nn.conv2d(X, con_weights, 
                             strides=[1,conv1_step,conv1_step,1], 
                             padding='VALID');
        conv_out=act_func(tf.nn.bias_add(conv,conv_biases));

    '''    

    '''    
    with tf.variable_scope('layer2-pool1',reuse=tf.AUTO_REUSE):
        pool_out = tf.nn.max_pool(conv_out, 
                                   ksize=[1,pool1_size,pool1_size,1],
                                    strides=[1,pool1_step,pool1_step,1], 
                                    padding='SAME');
    
    '''    

    '''
    with tf.variable_scope('layer3-conv2',reuse=tf.AUTO_REUSE):
        con_shape=[conv2_size,conv2_size,conv1_deep,conv2_deep];
        con_weights=get_weight_variable(con_shape, None);
        conv_biases=tf.get_variable('biases',[conv2_deep],
                                     initializer=tf.constant_initializer(0.0));
        conv = tf.nn.conv2d(pool_out, con_weights, 
                             strides=[1,conv2_step,conv2_step,1], 
                             padding='VALID');
        conv_out=act_func(tf.nn.bias_add(conv,conv_biases));   
        

    '''    

    '''    
    with tf.variable_scope('layer4-pool2',reuse=tf.AUTO_REUSE):
        pool_out = tf.nn.max_pool(conv_out, 
                                   ksize=[1,pool2_size,pool2_size,1],
                                    strides=[1,pool2_step,pool2_step,1], 
                                    padding='SAME');


    '''    

    '''
    with tf.variable_scope('layer5-conv3',reuse=tf.AUTO_REUSE):
        con_shape=[conv3_size,conv3_size,conv2_deep,conv3_deep];
        con_weights=get_weight_variable(con_shape, None);
        conv_biases=tf.get_variable('biases',[conv3_deep],
                                     initializer=tf.constant_initializer(0.0));
        conv = tf.nn.conv2d(pool_out, con_weights, 
                             strides=[1,conv3_step,conv3_step,1], 
                             padding='VALID');
        conv_out=act_func(tf.nn.bias_add(conv,conv_biases));




    tmp_shape = conv_out.get_shape().as_list();
    fc_input_size=tmp_shape[1]*tmp_shape[2]*tmp_shape[3];
    print('fc_input_size=',fc_input_size);
    fc_x = tf.reshape(conv_out,[-1,fc_input_size]);
    fc_layer_cot = len(fc_hiddens);
    
    with tf.variable_scope('layer6-fc1',reuse=tf.AUTO_REUSE):
        weights = get_weight_variable([fc_input_size,fc_hiddens[0]],regularizer);
        biase   =  tf.get_variable('biase1',
                                [fc_hiddens[0]],
                                initializer=tf.truncated_normal_initializer( stddev=0.05));
        layer   = act_func(tf.matmul(fc_x,weights)+biase);
    for lay_num in range(1,fc_layer_cot):
        var_name = 'layer%d-fc%d'%(lay_num+6,lay_num+1);
        with tf.variable_scope(var_name,reuse=tf.AUTO_REUSE):
            weights = get_weight_variable([fc_hiddens[lay_num-1],fc_hiddens[lay_num]],regularizer);
            biase   =  tf.get_variable('biase%d'%(lay_num+1),
                                [fc_hiddens[lay_num]],
                                initializer=tf.truncated_normal_initializer( stddev=0.05));
            layer   = act_func(tf.matmul(layer,weights)+biase);
    
    # 输出层设计
    with tf.variable_scope('out_layer',reuse=tf.AUTO_REUSE):
        weights = get_weight_variable([fc_hiddens[fc_layer_cot-1],label_size],regularizer);
        biase   =      tf.get_variable('biase_out',
                                [label_size],
                                initializer=tf.truncated_normal_initializer( stddev=0.05));
        layer   = tf.matmul(layer,weights)+biase;    
            
    return layer

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




########################## 训练部分 #######################


def train(datasource):
    X = tf.placeholder(tf.float32, [None,cnn_input_size,cnn_input_size,cnn_input_deep], 'X');
    Y = tf.placeholder(tf.float32, [None,label_size], 'Y');
    
    
    global_step = tf.Variable(0,trainable=False,name='gs');
    data_size = datasource.data_size();
    
    # 正则处理
    if need_regular: 
        regularizer = tf.contrib.layers.l2_regularizer(regular_lambda);
        py = get_inference(X, act_func, regularizer);
    else:
        py = get_inference(X, act_func, None);
    

    
    # 滑动平均
    if need_regular:
        variable_avage = tf.train.ExponentialMovingAverage(move_avage_rate,global_step);
        variable_avage_op = variable_avage.apply(tf.trainable_variables());
        
    # 损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=py));
    regloss = tf.get_collection('losses');
    if len(regloss)!=0:
        loss = loss + tf.add_n(regloss);
    # 递减学习率
    lr = tf.train.exponential_decay(learn_rate, global_step,
                                    data_size/batch_size,
                                    # 200,
                                    learn_rate_decy,
                                    staircase=True);
    
    # 优化训练过程
    train_step = tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_step);
    if need_regular:
        train_op = tf.group(train_step,variable_avage_op);
    else:
        train_op = train_step;
    
    saver = tf.train.Saver();
    with tf.Session() as sess:
        if load_value:
            saver.restore(sess, model_save_path);
        else:
            sess.run(tf.global_variables_initializer());
        now = time.time();
        for i in range(steps):
            start = (i*batch_size) % data_size;
            end = min(start+batch_size,data_size);
            xs,ys = datasource.getDataXY(start,end);
            xs = change_to_cnn_input(xs);
            _,lossv,pyv,yv,step = sess.run([train_op,loss,py,Y,global_step],{X:xs,Y:ys});
            if step%20==0:
                print('step=%d loss=%.5f time=%.2f'%(step,lossv,time.time()-now));
                now = time.time();
        saver.save(sess,model_save_path);
    print('finished!');                   


            
########################## 测评部分 #######################
    
def evel(datasource):
    X = tf.placeholder(tf.float32, [None,cnn_input_size,cnn_input_size,cnn_input_deep], 'X');
    py = get_inference(X, act_func,None);
    py = tf.nn.softmax(py, axis=1);
    saver = tf.train.Saver();
    with tf.Session() as sess:
        saver.restore(sess, model_save_path);
        xs,ys = datasource.getAllData();
        xs = change_to_cnn_input(xs);
        py = sess.run(py,{X:xs});
    
    data_size = datasource.data_size();
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
    py = get_inference(X, act_func,None);
    py = tf.nn.softmax(py, axis=1);
    saver = tf.train.Saver();
    with tf.Session() as sess:
        saver.restore(sess, model_save_path);
        xs = datasource.getAllDataX();
        xs_size= datasource.data_size();
        start =0;
        xs = change_to_cnn_input(xs);
        py_all=[];
        while start<xs_size:
            end = min(start+100,xs_size);
            # print(xs[start:end]);
            py_ = sess.run(py,{X:xs[start:end]});
            for i in py_:
                py_all.append(i); 
            start+=100;
    return py_all;




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
    if need_train:
        print('\n开始训练')
        train(train_ds,);
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