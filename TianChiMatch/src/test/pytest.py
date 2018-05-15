# -*- coding: utf-8 -*-
'''
Created on 2018年3月6日

@author: zwp
'''

import numpy as np;

# 特征数，输入张量的shape
feature_size = 4;
# 标签数，输出张量的shape
label_size = 4;

# 输入向量调整，尺寸
cnn_input_size=int(np.ceil(np.sqrt(feature_size)));
# 输入向量调整，深度
cnn_input_deep=1;

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


if __name__ == '__main__':

    
    x = [[1,2,3,4],
         [5,4,3,2]
        ]
    print(change_to_cnn_input(x));
    
    pass