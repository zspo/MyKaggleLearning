# -*- coding: utf-8 -*-
'''
Created on 2018年3月1日

@author: zwp
'''

import csv;
import random as rdm;
import numpy as np;

def t2ichange(dtype):
    if dtype == 'star':
        return 0;
    elif dtype == 'galaxy':
        return 1;
    elif dtype == 'qso':
        return 2;
    elif dtype=='unknown':
        return 3;
    else:
        return -1;

def i2tchange(itype):
    if      itype==0:return 'star';
    elif    itype==1:return 'galaxy';
    elif    itype==2:return 'qso';
    elif    itype==3:return 'unknown';
    return '-1';

def type2int(csv_path_in,csv_path_out):
    '''
        将csv 的index文件 type字段转换为整型
        star   -> 0
        galaxy -> 1
        qso    -> 2
        unknown-> 3
    '''
    f_in = open(csv_path_in,'r');
    f_out= open(csv_path_out,'w');
    origin = csv.reader(f_in);
    out = csv.writer(f_out);
    for line in origin:
        if origin.line_num == 1:
            out.writerow(line);
        else:
            out.writerow([line[0],t2ichange(line[1])]);
    f_in.close();
    f_out.close();
    print('type2int finished! outpaht=',csv_path_out);
    

def class_spliter(origin_path,out_pathes):
    '''
        将csv 的index文件 按照类型分割成4个csv文件,
        其中pathes的路径是依次是star(0),galaxy(1),qso(2),unknown(3)
    '''
    f_in = open(origin_path,'r');
    f_outs=[];
    f_outs.append(open(out_pathes[0],'w'));
    f_outs.append(open(out_pathes[1],'w'));
    f_outs.append(open(out_pathes[2],'w'));
    f_outs.append(open(out_pathes[3],'w'));
    
    origin = csv.reader(f_in);
    outs = [];
    for i in range(4):
        outs.append(csv.writer(f_outs[i]))

    for line in origin:
        if origin.line_num == 1:
            for i in range(4):
                outs[i].writerow(line);
        else:
            outs[int(line[1])].writerow(line);
    f_in.close();
    for f in f_outs:
        f.close();
    print('class_spliter finished outpath=',out_pathes);    

def randomint(a,b):
    return rdm.randint(a,b);
        
def random_spliter(origin_path,precent,target_path,left_path):
    prc = int(precent*1000);
    f_in = open(origin_path,'r');
    f_out_tg= open(target_path,'w');
    if not left_path is None:
        f_out_lf= open(left_path,'w');
    
    origin = csv.reader(f_in);
    out_tg = csv.writer(f_out_tg);
    if not left_path is None:
        out_lf = csv.writer(f_out_lf);
    
    cot_tg=0;
    cot_lf=0;
    for line in origin:
        if origin.line_num == 1:
            out_tg.writerow(line);
            if not left_path is None:
                out_lf.writerow(line);
        elif randomint(0,999)<prc:
            out_tg.writerow(line);
            cot_tg+=1;
        else:
            if not left_path is None:
                out_lf.writerow(line);
                cot_lf+=1;
    f_in.close();
    f_out_tg.close();
    if not left_path is None:
        f_out_lf.close();
    if cot_lf==0 and cot_tg==0:
        cot_lf=-1;
    print('random_spliter finished! target_cot=%d,left_cot=%d,prc=%.2f'%(cot_tg,cot_lf,cot_tg/(cot_tg+cot_lf)*100.0))   
    pass;


def get_rdlist_by_sizelist(sizes):
    sizes = np.array(sizes,dtype=np.float32);
    rdindex=sizes/np.sum(sizes)*1000;
    rdlist=[0];
    for i in range(1,rdindex.shape[0]):
        tmp = int(rdindex[i-1]+rdlist[i-1]);
        rdlist.append(tmp);
    rdlist.append(1000);
    return rdlist;

def get_random_id(rdlist):
    id = 0;
    size = len(rdlist);
    rid = randomint(0,999);
    for i in range(1,size):
        if rid <= rdlist[i]:
            return id;
        else:
            id+=1;
def csv_size(path_file):
    return sum(1 for line in open(path_file));
    
def random_merge(origin_pathes,out_path):
    f_cot = len(origin_pathes);
    merge_in_fs = [];
    merge_in_origin=[];
    flags=[];
    csv_sizes=[];
    cot=0;
    merge_out_f = open(out_path,'w');
    for i in range(f_cot):
        csv_sizes.append(csv_size(origin_pathes[i]));
        merge_in_fs.append(open(origin_pathes[i]));
        flags.append(True);
    merge_out = csv.writer(merge_out_f);
    merge_in_origin.append(csv.reader(merge_in_fs[0]));
    merge_out.writerow(next(merge_in_origin[0]));
    for i in range(1,f_cot):
        merge_in_origin.append(csv.reader(merge_in_fs[i]));
        next(merge_in_origin[i])
    
    rdlist=get_rdlist_by_sizelist(csv_sizes);
    
    while True:
        f = get_random_id(rdlist);
        if flags[f]:
            try:
                merge_out.writerow(next(merge_in_origin[f]));
                cot+=1;
            except StopIteration:
                flags[f]= False;           
        tf=True;
        for i in range(f_cot):
            if flags[i]:
                tf=False;
                continue;
        if tf:break;
        
    merge_out_f.close();    
    for i in range(f_cot):
        merge_in_fs[i].close();
    print('random_merge finished! cot=%d'%(cot));

def random_merge_for_uk_type(origin_pathes,out_path):
    f_cot = len(origin_pathes);
    merge_in_fs = [];
    merge_in_origin=[];
    flags=[];
    csv_sizes=[];
    cot=0;
    merge_out_f = open(out_path,'w');
    for i in range(f_cot):
        csv_sizes.append(csv_size(origin_pathes[i]));
        merge_in_fs.append(open(origin_pathes[i]));
        flags.append(True);
    merge_out = csv.writer(merge_out_f);
    merge_in_origin.append(csv.reader(merge_in_fs[0]));
    merge_out.writerow(next(merge_in_origin[0]));
    for i in range(1,f_cot):
        merge_in_origin.append(csv.reader(merge_in_fs[i]));
        next(merge_in_origin[i])
    
    rdlist=get_rdlist_by_sizelist(csv_sizes);
    
    while True:
        f = get_random_id(rdlist);
        if flags[f]:
            try:
                it = next(merge_in_origin[f]);
                if it[1]=='3' or it[1]==3:
                    merge_out.writerow(it);
                else:
                    it[1]='0';
                    merge_out.writerow(it);
                cot+=1;
            except StopIteration:
                flags[f]= False;           
        tf=True;
        for i in range(f_cot):
            if flags[i]:
                tf=False;
                continue;
        if tf:break;
        
    merge_out_f.close();    
    for i in range(f_cot):
        merge_in_fs[i].close();
    print('random_merge_for_uk_type finished! cot=%d'%(cot));


def random_merge_for_o3_type(origin_pathes,out_path):
    f_cot = len(origin_pathes);
    merge_in_fs = [];
    merge_in_origin=[];
    flags=[];
    csv_sizes=[];
    cot=0;
    merge_out_f = open(out_path,'w');
    for i in range(f_cot):
        csv_sizes.append(csv_size(origin_pathes[i]));
        merge_in_fs.append(open(origin_pathes[i]));
        flags.append(True);
    merge_out = csv.writer(merge_out_f);
    merge_in_origin.append(csv.reader(merge_in_fs[0]));
    merge_out.writerow(next(merge_in_origin[0]));
    for i in range(1,f_cot):
        merge_in_origin.append(csv.reader(merge_in_fs[i]));
        next(merge_in_origin[i])
    
    rdlist=get_rdlist_by_sizelist(csv_sizes);
    
    while True:
        f = get_random_id(rdlist);
        if flags[f]:
            try:
                it = next(merge_in_origin[f]);
                if not it[1] == '3' :
                    merge_out.writerow(it);
                    cot+=1;
            except StopIteration:
                flags[f]= False;           
        tf=True;
        for i in range(f_cot):
            if flags[i]:
                tf=False;
                continue;
        if tf:break;
        
    merge_out_f.close();    
    for i in range(f_cot):
        merge_in_fs[i].close();
    print('random_merge_for_o3_type finished! cot=%d'%(cot));




def create_result_csv(indexIds,py,out_path):
    f_out = open(out_path,'w');
    out=csv.writer(f_out);
    itypes=np.argmax(py,axis=1);
    print('max ',itypes);
    for i in range(len(indexIds)):
        out.writerow([indexIds[i],i2tchange(itypes[i])]);
    f_out.close();
    print('create_result_csv finished! out size=%d'%(len(indexIds)));
    pass;


################################################################        
base_path=r'/home/zwp/work/Dataset/tianci/LAMOST';

origin_path = base_path+r'/first_train_index_20180131.csv';
typechanged_out_path = base_path+r'/first_train_index_types.csv';
class_spilter_out_paths=[  base_path+r'/index_types_star.csv',
                           base_path+r'/index_types_galaxy.csv',
                           base_path+r'/index_types_qso.csv',
                           base_path+r'/index_types_unknown.csv'
                           ];

cut_target_set_out_paths=[ base_path+r'/index_types_star_tg.csv',
                           base_path+r'/index_types_galaxy_tg.csv',
                           base_path+r'/index_types_qso_tg.csv',
                           base_path+r'/index_types_unknown_tg.csv'];
cut_left_set_out_paths=  [ base_path+r'/index_types_star_lf.csv',
                           base_path+r'/index_types_galaxy_lf.csv',
                           base_path+r'/index_types_qso_lf.csv',
                           base_path+r'/index_types_unknown_lf.csv']

train_set_out_path = base_path+r'/index_train.csv';
test_set_out_path = base_path+r'/index_test.csv';

train_set_uk_out_path = base_path+r'/index_train_uk.csv';
test_set_uk_out_path = base_path+r'/index_test_uk.csv';

train_set_3o_out_path = base_path+r'/index_train_3o.csv';
test_set_3o_out_path = base_path+r'/index_test_3o.csv';

split_precent=[4000/442969.0, # star=6000 
               4000/5231.0, # galaxy=4700
               1300/1363.0, # qso=1220 * 3=3600
               3000/34288.0#unknown=1000
               ];

def run():
    type2int(origin_path,typechanged_out_path);
    class_spliter(typechanged_out_path, class_spilter_out_paths);
    for i in range(4):
        random_spliter(class_spilter_out_paths[i],split_precent[i],
                       cut_target_set_out_paths[i],cut_left_set_out_paths[i]);
    star_test_tmp_path=base_path+r'/star_test_tmp.csv';
    uk_test_tmp_paht=base_path+r'/uk_test_tmp.csv';
    train_tmp_path=base_path+r'/train_tmp1.csv';
    train_tmp2_path=base_path+r'/train_tmp2.csv';
                                       
    random_merge(cut_target_set_out_paths,train_tmp_path);#[6000,4700,1200,3000]
    random_merge([train_tmp_path,cut_target_set_out_paths[2]],train_tmp2_path);# add qso [6000,4700,2400,3000]
    random_merge([train_tmp2_path,cut_target_set_out_paths[2]],train_set_out_path);# add qso [6000,4700,3600,3000]
    
    random_merge_for_uk_type([train_tmp2_path,cut_target_set_out_paths[2]],train_set_uk_out_path);
    random_merge_for_o3_type([train_tmp2_path,cut_target_set_out_paths[2]],train_set_3o_out_path);
    
    random_spliter(class_spilter_out_paths[0],4000/442969.0,# 2000
                       star_test_tmp_path,None);# 
    random_spliter(cut_left_set_out_paths[3],1000/34288.0,# 1000
                       uk_test_tmp_paht,None);

    random_spliter(class_spilter_out_paths[2],1300/1363.0,# 2000
                       cut_left_set_out_paths[2],None);# 
                       
    random_spliter(cut_left_set_out_paths[1],0.99,# c从剩下的galaxy 中获取
                       train_tmp_path,None);#    
    
    test_paths=[star_test_tmp_path,
                train_tmp_path,
                cut_left_set_out_paths[2],
                uk_test_tmp_paht];
    random_merge(test_paths,test_set_out_path);
    random_merge_for_uk_type(test_paths,test_set_uk_out_path);
    random_merge_for_o3_type(test_paths,test_set_3o_out_path);
    
    pass;

def test():
    print(get_rdlist_by_sizelist([100,10,5,1]));
    pass;


if __name__ == '__main__':
    run();
    # test();
    pass