#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:23:45 2018

@author: hxj
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 20:32:28 2018

@author: hxj
"""

import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import _pickle as cPickle
#import base
#import sigproc
eps = 1e-5


def read_file(filename):
    file = wave.open(filename,'r')    
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype = np.short)
    #wavedata = np.float(wavedata*1.0/max(abs(wavedata)))  # normalization)
    time = np.arange(0,wav_length) * (1.0/framerate)
    file.close()
    return wavedata, time, framerate

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def zscore(data,mean,std):
    shape = np.array(data.shape,dtype = np.int32)
    for i in range(shape[0]):
        data[i,:,:,0] = (data[i,:,:,0]-mean)/(std)
    return data

def normalization(data):
    '''
    #apply zscore
    mean = np.mean(data,axis=0)#axis=0纵轴方向求均值
    std = np.std(data,axis=0)
    train_data = zscore(train_data,mean,std)
    test_data = zscore(test_data,mean,std)
    '''
    mean = np.mean(data,axis=0)#axis=0纵轴方向求均值
    std = np.std(data,axis=0)
    data = (data-mean)/std
    return data

def mapminmax(data):
    shape = np.array(data.shape,dtype = np.int32)
    for i in range(shape[0]):
        min = np.min(data[i,:,:,0])
        max = np.max(data[i,:,:,0])
        data[i,:,:,0] = (data[i,:,:,0] - min)/((max - min)+eps)
    return data

def generate_label(emotion,classnum):
    label = -1
    if(emotion == 'ang'):
        label = 0
    elif(emotion == 'sad'):
        label = 1
    elif(emotion == 'hap'):
        label = 2
    elif(emotion == 'neu'):
        label = 3
    elif(emotion == 'fear'):
        label = 4
    else:
        label = 5
    return label
        
        
def statistic():
    
    train_num = 2928
    filter_num = 20
    rootdir = '/home/jamhan/hxj/datasets/IEMOCAP_full_release'
    traindata1 = np.empty((train_num*300,filter_num),dtype=np.float32) #300是什么
    traindata2 = np.empty((train_num*300,filter_num),dtype=np.float32)
    traindata3 = np.empty((train_num*300,filter_num),dtype=np.float32)
    train_num = 0
    
    
    for speaker in os.listdir(rootdir):
        if(speaker[0] == 'S'):#文件夹名首字符是 S
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav') #wav路径
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation') #label路径
            for sess in os.listdir(sub_dir):
                if(sess[7] == 'i'): #只要即兴表演的
                    emotdir = emoevl+'/'+sess+'.txt' #对应label文件路径
                    #emotfile = open(emotdir)
                    emot_map = {}
                    with open(emotdir,'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if(line[0] == '['):
                                t = line.split()
                                emot_map[t[3]] = t[4] #依照顺序读对应的标签存到 emot_map 字典里{filename:'lable'}
                                
        
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir) #对应sess下所有.wav的路径 列表
                    for filename in files:
                        #wavname = filename[-23:-4]
                        wavname = filename.split("/")[-1][:-4] #wave name
                        emotion = emot_map[wavname] #对应wav的标签
                        if(emotion in ['hap','ang','neu','sad']):
                             data, time, rate = read_file(filename)
                             mel_spec = ps.logfbank(data, rate, nfilt=filter_num) #对整个wav算三组特征
                             delta1 = ps.delta(mel_spec, 2)
                             delta2 = ps.delta(delta1, 2)
                             
                             time = mel_spec.shape[0] 
                             if(speaker in ['Session1','Session2','Session3','Session4']):
                                 #training set
                                 if(time <= 300):
                                      part = mel_spec
                                      delta11 = delta1
                                      delta21 = delta2 #在尾部填0
                                      part = np.pad(part, ((0, 300 - part.shape[0]), (0,0)), 'constant',constant_values = 0)
                                      delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0,0)), 'constant',constant_values = 0)
                                      delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0,0)), 'constant',constant_values = 0)
                                      traindata1[train_num*300:(train_num+1)*300] = part
                                      traindata2[train_num*300:(train_num+1)*300] = delta11
                                      traindata3[train_num*300:(train_num+1)*300] = delta21
                                      
                                      em = generate_label(emotion,6)
                                      train_num = train_num + 1 #填充完的特征向量首尾相连塞到traindata里面
                                 else:
                                      
                                     if(emotion in ['ang','neu','sad']):
                                         
                                         for i in range(2):#??????????????
                                             if(i == 0):
                                                 begin = 0
                                                 end = begin + 300
                                             else:
                                                 begin = time - 300
                                                 end = time
                                          
                                             part = mel_spec[begin:end,:]
                                             delta11 = delta1[begin:end,:]
                                             delta21 = delta2[begin:end,:]
                                             traindata1[train_num*300:(train_num+1)*300] = part
                                             traindata2[train_num*300:(train_num+1)*300] = delta11
                                             traindata3[train_num*300:(train_num+1)*300] = delta21
                                             train_num = train_num + 1
                                     else:#长度大于3s的hap样本
                                        frames = divmod(time-300,100)[0] + 1 #帧数
                                        for i in range(frames):
                                            begin = 100*i
                                            end = begin + 300
                                            part = mel_spec[begin:end,:]
                                            delta11 = delta1[begin:end,:]
                                            delta21 = delta2[begin:end,:]
                                            traindata1[train_num*300:(train_num+1)*300] = part
                                            traindata2[train_num*300:(train_num+1)*300] = delta11
                                            traindata3[train_num*300:(train_num+1)*300] = delta21
                                            train_num = train_num + 1
                                          
                             else:
                                 pass
                                    
                                 
                        else:
                            pass
    
    
        mean1 = np.mean(traindata1,axis=0)#axis=0纵轴方向求均值 #每个样本之间计算
        std1 = np.std(traindata1,axis=0)
        mean2 = np.mean(traindata2,axis=0)#axis=0纵轴方向求均值
        std2 = np.std(traindata2,axis=0)
        mean3 = np.mean(traindata3,axis=0)#axis=0纵轴方向求均值
        std3 = np.std(traindata3,axis=0)
        output = './zscore'+str(filter_num)+'.pkl'
        #output = './IEMOCAP'+str(m)+'_'+str(filter_num)+'.pkl'
        f=open(output,'wb')
        cPickle.dump((mean1,std1,mean2,std2,mean3,std3),f) #费劲巴拉算了一堆参数
        f.close()           
    return
                
        


if __name__=='__main__':
    read_CASIA()
    #print "test_num:", test_num
    #print "train_num:", train_num
#    n = wgn(x, 6)
#    xn = x+n # 增加了6dBz信噪比噪声的信号
