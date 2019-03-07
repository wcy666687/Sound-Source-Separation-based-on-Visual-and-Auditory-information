import data_create
import os
import numpy as np
from keras.models import load_model
def seperation_single():
    path='gt_audio\\accordion_3_acoustic_guitar_3.wav'

    #分离音频中的nmf特征，其中将幅度和相位分离，P是相位
    V1_list,V2_list,res,stft_out=data_create.create_nmf_features_develop(path,200,128)
    #V1_list, V2_list=data_create.load_model_to_check_bin(W,H,path)
    #载入已经训练好的模型进行分离
    print('1')
    #V1_list, V2_list = data_create.use_kmeans_to_check(W, H, path)
    #V1_list, V2_list =data_create.load_model_to_check(W,H,path,8)
    #V1_list, V2_list=data_create.sort_to_d(W,H,path,20)
    print('2')
    #V1,V2=data_create.comb_V(V1_list,V2_list)
    #将分离好的幅度和相位重建信号谱图
    print('3')
    #v_1=data_create.reconstruct_abs_phase_res(V1,P,res)
    print('4')
    #v_2 = data_create.reconstruct_abs_phase_res(V2, P, res)
    v_1,v_2=data_create.ming_zi_zhen_nan_qu(V1_list,V2_list,stft_out,res)
    x,y=np.shape(v_1)
    resz=np.zeros((1,y))
    v_1=np.vstack((v_1,resz))
    v_2=np.vstack((v_2,resz))
    #写入
    data_create.write_to_file(v_1, '1_test.wav')
    data_create.write_to_file(v_2, '2_test.wav')

def seperate(path):

    file_list=os.listdir(path)
    for i in file_list:
        real_path=path+i
        #W, H, P, res, stft_out = data_create.create_nmf_features(real_path, 20, 200)
        V1_list, V2_list, res, stft_out = data_create.create_nmf_features_develop(real_path, 200, 128)
        #V1_list, V2_list = data_create.load_model_to_check(W, H, real_path, 16)
        #V1,V2=data_create.comb_V(V1_list,V2_list)
        #v_1 = data_create.reconstruct_abs_phase_res(V1, P, res)
        #v_2 = data_create.reconstruct_abs_phase_res(V2, P, res)

        #V1_list, V2_list = data_create.sort_to_d(W, H, real_path, 20)
        v_1, v_2 = data_create.ming_zi_zhen_nan_qu(V1_list, V2_list, stft_out, res)
        x, y = np.shape(v_1)
        resz = np.zeros((1, y))
        v_1 = np.vstack((v_1, resz))
        v_2 = np.vstack((v_2, resz))
        i=i[:-4]
        data_create.write_to_file_develop(v_1, 'result_audio\\'+i+'_seg1'+'.wav')
        data_create.write_to_file_develop(v_2, 'result_audio\\'+i+'_seg2'+'.wav')



if __name__=='__main__':
    path = 'gt_audio\\'
    seperate(path)