import wave
import numpy as np
import python_speech_features as ps
import os
import z_satistic
import glob
import _pickle as cPickle

train_dataset_region = ['Session1', 'Session2', 'Session3', 'Session4']
test_dataset_region = ['Session5']
filter_num = 40
total_train_sample_num = 2817
train_nummap = {'hap': 434, 'ang': 631, 'neu': 1045, 'sad': 707}
total_test_sample_num = 898
test_nummap = {'hap': 240, 'ang': 92, 'neu': 376, 'sad': 190}
sample_length = 300
sample_overlap = 100
dataset_length = min(train_nummap['hap'], train_nummap['sad'], train_nummap['ang'], train_nummap['neu'])

def read_file(filename):
    file = wave.open(filename, 'r')
    file_params = file.getparams()
    nchannels, sampwidth, framerate, frame_num = file_params[:4]
    file_str = file.readframes(frame_num)
    file_data = np.fromstring(file_str,dtype=np.short)
    file_time = np.arange(0,frame_num) * (1.0/framerate)
    file.close()
    return file_data, file_time, framerate

def load_file():
    f = open('z_score.pkl', 'rb')
    mean, stdv = cPickle.load(f)
    return mean, stdv

def generate_label(emotion):
    if(emotion == 'ang'):
        label = 0
    elif(emotion == 'sad'):
        label = 1
    elif(emotion == 'hap'):
        label = 2
    elif(emotion == 'neu'):
        label = 3
    else:
        label = 4
    return label

def generate_dataset():
    dataset_path = 'D:\IEMOCAP\IEMOCAP_full_release'

    train_dataset = np.empty((total_train_sample_num, sample_length, filter_num, 1))
    test_dataset = np.empty((total_test_sample_num, sample_length, filter_num, 1))
    train_label = np.empty(total_train_sample_num)
    test_label = np.empty(total_test_sample_num)
    Test_label = np.empty((total_test_sample_num, 1), dtype=np.int8)

    IEMOCAP_mean, IEMOCAP_stdv = load_file()


    sample_num = 0
    for session in os.listdir(dataset_path):
        if(session in train_dataset_region):
            dialog_path = os.path.join(dataset_path, session, 'sentences/wav')
            label_path = os.path.join(dataset_path, session, 'dialog/EmoEvaluation')
            for dialog_name in os.listdir(dialog_path):
                if dialog_name[7]=='i':
                    dialog_label_name = label_path+'/'+dialog_name+'.txt'
                    filename_label_map = {}
                    with open(dialog_label_name, 'r') as read_dialog_label:
                        while True:
                            line = read_dialog_label.readline()
                            if not line:
                                break
                            if line[0]=='[':
                                t = line.split()
                                filename_label_map[t[3]] = t[4]
                    read_dialog_label.close()

                    waves_path = os.path.join(dialog_path, dialog_name)
                    waves = os.listdir(waves_path)
                    for file_name in waves:
                        wave_name = file_name.split("/")[-1][:-4]
                        wave_path = os.path.join(waves_path, file_name)
                        emotion = filename_label_map[wave_name]
                        if(emotion in ['hap','ang','neu','sad']):
                            wave_data, wave_time, wave_framerate = read_file(wave_path)
                            mel_spec = ps.logfbank(wave_data, wave_framerate, nfilt=filter_num)
                            frame_num = mel_spec.shape[0]
                            emotion_label = generate_label(emotion)

                            if (emotion in ['hap','ang']):
                                if (frame_num < sample_length):
                                    mel_data = (mel_spec-IEMOCAP_mean)/IEMOCAP_stdv
                                    mel_data = np.pad(mel_data, ((0, sample_length-mel_spec.shape[0]), (0, 0)),'constant', constant_values=0)
                                    train_dataset[sample_num, :, :, 0] = mel_data
                                    train_label[sample_num] = emotion_label
                                    sample_num = sample_num + 1
                                    print(session, sample_num*100/total_train_sample_num)

                                else:
                                    mel_data = (mel_spec - IEMOCAP_mean) / IEMOCAP_stdv
                                    wave_sample_num = (frame_num-sample_length) // sample_overlap + 1
                                    for i in range(wave_sample_num):
                                        begin = sample_overlap*i
                                        end = sample_length + sample_overlap*i
                                        train_dataset[sample_num, :, :, 0] = mel_data[begin:end]
                                        train_label[sample_num] = emotion_label
                                        sample_num = sample_num+1
                                    #sample_num = sample_num + wave_sample_num
                                    print(session, sample_num*100/total_train_sample_num, )

                            else:
                                if (frame_num < sample_length):
                                    mel_data = (mel_spec - IEMOCAP_mean) / IEMOCAP_stdv
                                    mel_data = np.pad(mel_data, ((0, sample_length-mel_spec.shape[0]), (0, 0)),'constant', constant_values=0)
                                    train_dataset[sample_num, :, :, 0] = mel_data
                                    train_label[sample_num] = emotion_label
                                    sample_num = sample_num + 1
                                    print(session, sample_num*100/total_train_sample_num)
                                else:
                                    wave_sample_num = frame_num // sample_length
                                    mel_data = (mel_spec - IEMOCAP_mean) / IEMOCAP_stdv
                                    for sample_num_wav in range(wave_sample_num):
                                        begin = sample_length*sample_num_wav
                                        end = begin + sample_length
                                        train_dataset[sample_num, :, :, 0] = mel_data[begin:end]
                                        train_label[sample_num] = emotion_label
                                        sample_num = sample_num + 1
                                    #sample_num = sample_num + wave_sample_num
                                    print(session, sample_num*100/total_train_sample_num)
                        else:
                            pass

    test_num = 0
    for session in os.listdir(dataset_path):
        if(session in test_dataset_region):
            dialog_path = os.path.join(dataset_path, session, 'sentences/wav')
            label_path = os.path.join(dataset_path, session, 'dialog/EmoEvaluation')
            for dialog_name in os.listdir(dialog_path):
                if dialog_name[7]=='i':
                    dialog_label_name = label_path+'/'+dialog_name+'.txt'
                    filename_label_map = {}
                    with open(dialog_label_name, 'r') as read_dialog_label:
                        while True:
                            line = read_dialog_label.readline()
                            if not line:
                                break
                            if line[0]=='[':
                                t = line.split()
                                filename_label_map[t[3]]=t[4]
                    read_dialog_label.close()

                    waves_path = os.path.join(dialog_path, dialog_name)
                    waves = os.listdir(waves_path)
                    for file_name in waves:
                        wave_name = file_name.split("/")[-1][:-4]
                        wave_path = os.path.join(waves_path, file_name)
                        emotion = filename_label_map[wave_name]
                        if(emotion in ['hap','ang','neu','sad']):
                            wave_data, wave_time, wave_framerate = read_file(wave_path)
                            mel_spec = ps.logfbank(wave_data, wave_framerate, nfilt=filter_num)
                            frame_num = mel_spec.shape[0]
                            emotion_label = generate_label(emotion)

                            if (emotion in ['hap','ang']):
                                if (frame_num < sample_length):
                                    mel_data = (mel_spec-IEMOCAP_mean)/IEMOCAP_stdv
                                    mel_data = np.pad(mel_data, ((0, sample_length-mel_spec.shape[0]), (0, 0)),'constant', constant_values=0)
                                    test_dataset[test_num, :, :, 0] = mel_data
                                    test_label[test_num] = emotion_label
                                    test_num = test_num + 1
                                    print('测试集', test_num *100/total_test_sample_num)

                                else:
                                    mel_data = (mel_spec - IEMOCAP_mean) / IEMOCAP_stdv
                                    wave_sample_num = (frame_num-sample_length) // sample_overlap + 1
                                    for i in range(wave_sample_num):
                                        begin = sample_overlap*i
                                        end = sample_length + sample_overlap*i
                                        test_dataset[test_num , :, :, 0] = mel_data[begin:end]
                                        test_label[test_num ] = emotion_label
                                        test_num = test_num +1
                                    #sample_num = sample_num + wave_sample_num
                                    print('测试集', test_num *100/total_test_sample_num)

                            else:
                                if (frame_num < sample_length):
                                    mel_data = (mel_spec - IEMOCAP_mean) / IEMOCAP_stdv
                                    mel_data = np.pad(mel_data, ((0, sample_length-mel_spec.shape[0]), (0, 0)),'constant', constant_values=0)
                                    test_dataset[test_num, :, :, 0] = mel_data
                                    test_label[test_num] = emotion_label
                                    test_num = test_num + 1
                                    print('测试集', test_num *100/total_test_sample_num)
                                else:
                                    wave_sample_num = frame_num // sample_length
                                    mel_data = (mel_spec - IEMOCAP_mean) / IEMOCAP_stdv
                                    for sample_num_wav in range(wave_sample_num):
                                        begin = sample_length*sample_num_wav
                                        end = begin + sample_length
                                        test_dataset[test_num, :, :, 0] = mel_data[begin:end]
                                        test_label[test_num] = emotion_label
                                        test_num = test_num +1
                                    #sample_num = sample_num + wave_sample_num
                                    print('测试集', test_num *100/total_test_sample_num)
                        else:
                            pass

    hap_loca = np.arange(train_nummap['hap'])
    ang_loca = np.arange(train_nummap['ang'])
    neu_loca = np.arange(train_nummap['neu'])
    sad_loca = np.arange(train_nummap['sad'])

    hap_i = 0
    ang_i = 0
    neu_i = 0
    sad_i = 0

    for l in range(total_train_sample_num):
        if(train_label[l] == 0): #每个类型样本在traindata中的位置
            ang_loca[ang_i] = l
            ang_i = ang_i + 1
        elif (train_label[l] == 1):
            sad_loca[sad_i] = l
            sad_i = sad_i + 1
        elif (train_label[l] == 2):
            hap_loca[hap_i] = l
            hap_i = hap_i + 1
        else:
            neu_loca[neu_i] = l
            neu_i = neu_i + 1

    random_hap = np.random.permutation(hap_i)
    random_ang = np.random.permutation(ang_i)
    random_sad = np.random.permutation(sad_i)
    random_neu = np.random.permutation(neu_i)

    hap_location = hap_loca[random_hap]
    ang_location = ang_loca[random_ang]
    sad_location = sad_loca[random_sad]
    neu_location = neu_loca[random_neu]

    hap_data = train_dataset[hap_location[0:dataset_length], :, :, 0]
    ang_data = train_dataset[ang_location[0:dataset_length], :, :, 0]
    sad_data = train_dataset[sad_location[0:dataset_length], :, :, 0]
    neu_data = train_dataset[neu_location[0:dataset_length], :, :, 0]

    Train_dataset = np.empty([4*dataset_length, sample_length, filter_num, 1], dtype=np.float32)
    Train_label = np.empty([4*dataset_length, 1], dtype=np.int8)

    Train_dataset[0:dataset_length, :, :, 0] = hap_data
    Train_label[0:dataset_length, :] = 2
    Train_dataset[dataset_length:2*dataset_length, :, :, 0] = ang_data
    Train_label[dataset_length:2*dataset_length, :] = 0
    Train_dataset[2*dataset_length:3*dataset_length, :, :, 0] = sad_data
    Train_label[2*dataset_length:3*dataset_length, :] = 1
    Train_dataset[3*dataset_length:4*dataset_length, :, :, 0] = neu_data
    Train_label[3*dataset_length:4*dataset_length, :] = 3

    location = np.random.permutation(4*dataset_length)
    print(location)
    Train_dataset = Train_dataset[location[0:], :, :, :]
    Train_label = Train_label[location[0:], 0]

    for i in range(total_test_sample_num):
        Test_label[i, 0] = test_label[i]

    print('Train_dataset.shape = ', Train_dataset.shape)
    output = '.IEMOCAP_Mel.pkl'
    f = open(output, 'wb')
    cPickle.dump((Train_dataset, Train_label, test_dataset, Test_label), f)
    f.close()
    return

if __name__=='__main__':
    generate_dataset()