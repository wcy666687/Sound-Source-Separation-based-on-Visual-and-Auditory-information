import os
import librosa
import numpy as np
from dataHelper.nussl.core.audio_signal import AudioSignal
import sklearn
from sklearn.cluster import KMeans
from keras.models import load_model
from scipy.fftpack.realtransforms import dct

def create_nmf_features(path,n_sources,t):
    #不仅仅是用于训练，还用于重建，所以相位phase和残留的矩阵res都要保留
    audio=AudioSignal(path)
    audio.stft(window_length=1024,hop_length=512,n_fft_bins=1023)
    stft=audio.stft_data
    x,y,_=np.shape(stft)
    n = int(y / t)
    stft = stft.reshape(x, y)


    res=stft[:,n*t:y]
    stft=stft[:,:n*t]
    stft_out = stft
# 如果效果不好，可以在此处进行拆分，使每次nmf的矩阵较小，分解效果可能会比较好
    phase=np.zeros((x,n*t))
    for i in range(x):
        for j in range(n*t):
            phase[i][j]=np.arctan2(stft[i][j].imag,stft[i][j].real)

    print("Computing approximations")
    stft=abs(stft)
    W=[]
    H=[]
    for i in range(n):
        stft_tmp=stft[:,i*t:(i+1)*t]
        transformer = sklearn.decomposition.NMF(n_components=n_sources)
        w = transformer.fit_transform(stft_tmp)
        h = transformer.components_
        W.append(w)
        H.append(h)
    return W,H,phase,res,stft_out


def create_nmf_features_develop(path, n_sources, t):
    # 不仅仅是用于训练，还用于重建，所以相位phase和残留的矩阵res都要保留
    genres = ['accordion', 'acoustic_guitar', 'cello', 'flute', 'saxophone', 'trumpet', 'violin', 'xylophone']
    audio = AudioSignal(path)
    x_type,y_type=type_verify(path)
    x_midi=np.load('data\\midi\\'+genres[x_type]+'.npy')
    y_midi = np.load('data\\midi\\' + genres[y_type] + '.npy')
    mix_midi=np.vstack((x_midi,y_midi))
    audio.stft(n_fft_bins=2047)
    stft = audio.stft_data
    x, y, _ = np.shape(stft)
    n = int(y / t)
    stft = stft.reshape(x, y)
    res = stft[:, n * t:y]
    stft = stft[:, :n * t]
    stft_out = stft
    # 如果效果不好，可以在此处进行拆分，使每次nmf的矩阵较小，分解效果可能会比较好
    print("Computing approximations")
    stft = abs(stft)
    V1 = []
    V2 = []
    for i in range(n):
        stft_tmp = stft[:, i * t:(i + 1) * t]
        H_sep, W_sep, n_iter = sklearn.decomposition.non_negative_factorization(
                               X=stft_tmp.T, W=None, H=mix_midi, n_components=n_sources,
                               init='random', update_H=False, solver='mu',
                               beta_loss='frobenius', tol=1e-4,
                               max_iter=2000, alpha=0., l1_ratio=0.,
                               regularization=None, random_state=None,
                               verbose=0, shuffle=False)
        H_x=H_sep[:,:int(n_sources/2)]
        H_y=H_sep[:,int(n_sources/2):]
        v_x=np.dot(x_midi.T,H_x.T)
        v_y = np.dot(x_midi.T, H_y.T)
        V1.append(v_x)
        V2.append(v_y)

    return V1, V2, res, stft_out


def real_to_complex(a,b):
    real=a*np.cos(b)
    imag=a*np.sin(b)
    return real + (imag * 1j)

def reconstruct_abs_phase_res(V,phase,res):
    #由分离好的谱和相位和残差重建
    x,y=np.shape(V)
    tmp=np.zeros((x,y),dtype=np.complex128)
    tmp_1,tmp_2=np.shape(phase)
    if x!=tmp_1:
        p=phase.T
    else:
        p=phase
    for j in range(x):
        for k in range(y):
            tmp[j][k]=real_to_complex(V[j][k],p[j][k])
    tmp=np.hstack((tmp,res))
    return tmp

def reconstruct_abs_phase(V,phase):
    #由分离好的谱和相位重建
    x,y=np.shape(V)
    tmp=np.zeros((x,y),dtype=np.complex128)
    tmp_1,tmp_2=np.shape(phase)
    if x!=tmp_1:
        p=phase.T
    else:
        p=phase
    for j in range(x):
        for k in range(y):
            tmp[j][k]=real_to_complex(V[j][k],p[j][k])
    return tmp

def wrap_kmeans(W ):
    #一个kmeans的封装，曾经用过来对nmf聚类分离
    km = KMeans(n_clusters=2)
    W_test=W.T
    km.fit(W_test)
    predict=km.predict(W_test)
    return predict

def write_to_file(V,filename):
    output=AudioSignal(stft=V)
    output.istft(window_length=1024,hop_length=512)
    output.write_audio_to_file(filename)

def write_to_file_develop(V,filename):
    output=AudioSignal(stft=V)
    output.istft()
    output.write_audio_to_file(filename)

def load_model_to_check(W,H,path,n_source):
    #载入模型来对nmf特征向量进行预测,输出是两个列表，分别是W和H矩阵按时间排列
    model=load_model('16_16_2.model')
    x,y=type_verify(path)
    n=len(W)
    V1_out=[]
    V2_out=[]
    for i in range(n):
        #n是时间上的
        W_tmp=W[i]
        H_tmp=H[i]
        l1=[]
        l2=[]
        for k in range(n_source):
            #n_source要和生成的数量对齐
            w_test=W_tmp[:,k:k+1]
            w_test=w_test.reshape(1,16,16,2)
            p_tmp=model.predict(w_test)
            if p_tmp[0][x]>p_tmp[0][y]:
                l1.append(k)
            else:
                l2.append(k)
        n1=len(l1)
        n2=len(l2)
        l1=[]
        l2=[]
        for k in range(n_source):
            # n_source要和生成的数量对齐
            w_test = W_tmp[:, k:k + 1]
            w_test = w_test.reshape(1, 16, 16, 2)
            p_tmp = model.predict(w_test)
            if n1-n2>=2:
                if p_tmp[0][y]==max(p_tmp[0]):
                    l2.append(k)
                else:
                    l1.append(k)
            elif n2-n1>=2:
                if p_tmp[0][x] == max(p_tmp[0]):
                    l1.append(k)
                else:
                    l2.append(k)
            else:
                if p_tmp[0][x] > p_tmp[0][y]:
                    l1.append(k)
                else:
                    l2.append(k)


        W_tmp_1 = W_tmp.T[l1].T
        W_tmp_2 = W_tmp.T[l2].T
        H_tmp_1 = H_tmp[l1]
        H_tmp_2 = H_tmp[l2]
        v_tmp_1=np.dot(W_tmp_1,H_tmp_1)
        v_tmp_2=np.dot(W_tmp_2,H_tmp_2)
        V1_out.append(v_tmp_1)
        V2_out.append(v_tmp_2)


            #l1.append(k)
            #l2.append(p_tmp[0][x]-p_tmp[0][y])
            #d=dict(zip(l1,l2))
    return V1_out, V2_out
    #W_1=W_tmp[result_1]
    #W_2=W_tmp[result_2]
    #i, j = np.shape(H)
    #if i>j:
    #    H_tmp = H.T
    #else:
    #    H_tmp = H
    #H_1=H_tmp[result_1]
    #H_2=H_tmp[result_2]
    #V_1=np.dot(W_1.T,H_1)
    #V_2=np.dot(W_2.T,H_2)
    #注意输出是source1和source交替输出，和reconstruct_list_V相结合

def sort_to_d(W,H,path,n_source):
    model = load_model('16_16_2.model')
    x, y = type_verify(path)
    n = len(W)
    V1_out = []
    V2_out = []
    for i in range(n):
        W_tmp = W[i]
        H_tmp = H[i]
        l1 = []
        l2 = []
        for k in range(n_source):

            w=W_tmp[:,k:k+1]
            w = w.reshape(1, 16, 16, 2)
            p_tmp = model.predict(w)
            l1.append(k)
            l2.append(p_tmp[0][x]-p_tmp[0][y])
            d=dict(zip(l1,l2))
        l1,l2=from_d_to_list(d)
        W_tmp_1 = W_tmp.T[l1].T
        W_tmp_2 = W_tmp.T[l2].T
        H_tmp_1 = H_tmp[l1]
        H_tmp_2 = H_tmp[l2]
        v_tmp_1 = np.dot(W_tmp_1, H_tmp_1)
        v_tmp_2 = np.dot(W_tmp_2, H_tmp_2)
        V1_out.append(v_tmp_1)
        V2_out.append(v_tmp_2)
    return V1_out,V2_out

def zero_pos(l):
    n=len(l)
    out=-1
    for i in range(n):
        if l[i]>0:
            out=i
            break
        else:
            pass
    return out


def use_kmeans_to_check(W,H,path):
    #载入模型来对nmf特征向量进行预测,输出是两个列表，分别是W和H矩阵按时间排列
    x,y=type_verify(path)
    n=len(W)
    V1_out=[]
    V2_out=[]
    for i in range(n):
        #n是时间上的
        W_tmp=W[i]
        H_tmp=H[i]
        prdict=wrap_kmeans(W_tmp)
        l1,l2=calculate(list(prdict))
        l1=list(l1)
        l2=list(l2)
        W_tmp_1 = W_tmp.T[l1].T
        W_tmp_2 = W_tmp.T[l2].T
        H_tmp_1 = H_tmp[l1]
        H_tmp_2 = H_tmp[l2]
        v_tmp_1 = np.dot(W_tmp_1, H_tmp_1)
        v_tmp_2 = np.dot(W_tmp_2, H_tmp_2)
        V1_out.append(v_tmp_1)
        V2_out.append(v_tmp_2)
    return V1_out, V2_out

def load_model_to_check_bin(W,H,path):
    #载入模型来对nmf特征向量进行预测,输出是两个列表，分别是W和H矩阵按时间排列
    model=load_model('16_16_2.model')
    x,y=type_verify(path)
    n=len(W)
    V1=[]
    V2=[]
    for i in range(n):
        W_tmp=W[i]
        H_tmp=H[i]
        W1=W_tmp[:,:1]
        W2=W_tmp[:,1:2]
        a,b=np.shape(W1)
        H1 = H_tmp[:1, :]
        H2 = H_tmp[1:2, :]
        w_test1=W1
        w_test2=W2
        w_test1=w_test1.reshape(1,16,16,2)
        w_test2=w_test2.reshape(1,16,16,2)
        p1=model.predict(w_test1)
        p2=model.predict(w_test2)
        if p1[0][x]==max(p1[0]):
            if p2[0][x]==max(p2[0]):
                v1=np.dot(W_tmp,H_tmp)
                w=np.zeros((a,b))
                v2=np.dot(w,H1)
                V1.append(v1)
                V2.append(v2)
            else:
                v1 = np.dot(W1, H1)
                v2=np.dot(W2,H2)
                V1.append(v1)
                V2.append(v2)
        elif p2[0][x]==max(p2[0]):
            v1 = np.dot(W2, H2)
            v2 = np.dot(W1, H1)
            V1.append(v1)
            V2.append(v2)
        elif (p2[0][y]==max(p2[0])) & (p1[0][y]==max(p1[0])):
            v2 = np.dot(W_tmp, H_tmp)
            w = np.zeros((a, b))
            v1 = np.dot(w, H_tmp)
            V1.append(v1)
            V2.append(v2)
        else:
            if p2[0][y]>p1[0][y]:
                v1 = np.dot(W2, H2)
                v2 = np.dot(W1, H1)
                V1.append(v1)
                V2.append(v2)
            else:
                v2 = np.dot(W2, H2)
                v1 = np.dot(W1, H1)
                V1.append(v1)
                V2.append(v2)
    return V1,V2











    return V1_out, V2_out


def load_mfcc_model_to_check(W,H,path):

    #一次小尝试，尝试用mfcc取预测和分类，做音源分离，由于mfcc要做stft处理，对得到的谱图去除相位因素
    # 所以可以尝试对每一行W和每一列H做矩阵乘法，得到幅度谱，再计算出相应的mfcc特征值，放入mfcc网络模型中做预测
    model = load_model('mfcc.model')
    x, yy = type_verify(path)
    i, j = np.shape(W)

    if i < j:
        W_tmp = W.T
        i, j = np.shape(W_tmp)
    else:
        W_tmp = W
    m, n = np.shape(H)
    if m==j:
        H_tmp=H
    else:
        H_tmp = H.T
        m, n = np.shape(H_tmp)
    l=[]
    Z_l=[]
    for i in range(j):
        w=W_tmp[:,i:i+1]
        h=H_tmp[i:i+1,:]
        v=np.dot(w,h)
        #if np.shape(v)!=np.shape(phase):
        #    p=phase.T
        #else:
        #    p=phase
        #v_test=reconstruct_abs_phase(v,p)
        #t_test=AudioSignal(stft=v_test)
        #t_test.istft(window_length=1024,hop_length=512)
        #z=t_test.audio_data
        #z=z.reshape(len(z.T),)
        #mfcc = librosa.feature.mfcc(z)
        mfcc=from_spec_to_mel(v)
        Z_tmp=calculate_prediction(mfcc)
        Z_l.append(Z_tmp)
        a, b = np.shape(mfcc)
        n = int(b / 20)
        mfcc = mfcc[:, :n*20]
        mfcc = mfcc.T
        mfcc=mfcc.reshape(n,20,20,1)
        p_x=0
        p_y=0
        for i_tmp in range(n):
            m=mfcc[i:i+1]
            p=model.predict(m)
            p_x_tmp=p[0][x]
            p_y_tmp=p[0][yy]
            p_x=p_x+p_x_tmp
            p_y=p_y+p_y_tmp
        p_x=p_x/n
        p_y=p_y/n
        if p_x>p_y:
            l.append(0)
        else:
            l.append(1)
    return  l,Z_l


def type_verify(path):
    #从名字中提取出两种乐器
    (filepath, filename) = os.path.split(path)
    x=-1
    y=-1
    genres = ['accordion', 'acoustic_guitar', 'cello', 'flute', 'saxophone', 'trumpet', 'violin', 'xylophone']
    for i in genres:
        result = i in filename
        if result==True:
            pos = genres.index(i)
            if x<0:
                x=pos
            else:
                y=pos
                break
    return x,y

def calculate_prediction(mfcc):
    a, b = np.shape(mfcc)
    n = int(b / 20)
    mfcc = mfcc[:, :n * 20]
    mfcc = mfcc.T
    mfcc = mfcc.reshape(n, 20, 20, 1)
    model = load_model('mfcc.model')
    p = model.predict(mfcc)
    Z = np.zeros((1, 8))
    for i in range(n):
        Z = Z + p[i:i + 1]
    Z=Z/n
    return Z

def calculate(list1):
    #统计聚类后的列数,用于0,1序列
    L1 = len(list1)  # 列表list1的长度
    list2 = list(set(list1))  # 可以用set，直接去掉重复的元素
    list2.sort(reverse=False)  # 将列表由小到大排序
    x=[]
    for m in range(2):
        X = set()  # 设定一个空的集合，用来存放这个元素的所在的位置
        start = list1.index(list2[m])
        for n in range(L1):
            stop = L1
            if list2[m] in tuple(list1)[start:stop]:
                a = list1.index(list2[m], start, stop)
                X.add(a)
                start = start + 1
        x.append(X)
    return x

def from_d_to_list(d):
    dd=sort_by_value(d)
    n=int(len(dd)/2)
    pos=zero_pos(dd)
    if abs(n-pos)<=3:
        l2=dd[:pos]
        l1=dd[pos:len(dd)]
    else:
        if n<pos:
            l2=dd[:n+2]
            l1=dd[n+2:len(dd)]
        else:
            l2 = dd[:n-2]
            l1 = dd[n - 2:len(dd)]
    return l1,l2



def reconstruct_list_V(W,H):
    '''
    由一列W和一列H得到一列V，这一列是按时间的排序,和函数load_model_to_check的输出相结合，列表中交替的是source1和source2
    W,H是list
    '''

    V1_out=[]
    V2_out=[]
    n=int(len(W)/2)
    for i in range(n):
        w1=W[i*2]
        w2=W[i*2+1]
        h1=H[i*2]
        h2=H[i*2+1]
        v1=np.dot(w1,h1)
        v2=np.dot(w2,h2)
        V1_out.append(v1)
        V2_out.append(v2)
    return V1_out,V2_out

def comb_V(V1,V2):
    #将一列V按时间顺序连接起来
    n=len(V1)
    v1 = V1[0]
    v2 = V2[0]
    v1_out = v1
    v2_out = v2
    for i in range(n-1):
        v1=V1[i+1]
        v2=V2[i+1]
        v1_out=np.hstack((v1_out, v1))
        v2_out = np.hstack((v2_out, v2))
    return v1_out,v2_out


def from_list_comb_V(W,H,l1,l2):
    #对于一段音频中的一个W和H，进行预测后聚类得到l1和l2，从中分离
    w=W
    h=H
    a,b=np.shape(w)
    if a>b:
        w=w.T
    a, b = np.shape(h)
    if a>b:
        h=h.T
    w_1=w[l1]
    w_2=w[l2]
    h_1=h[l1]
    h_2=h[l2]
    v_1=np.dot(w_1.T,h_1)
    v_2=np.dot(w_2.T,h_2)
    return v_1,v_2

def sort_by_value(d):
    #排序
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort()
    return [ backitems[i][1] for i in range(0,len(backitems))]

def from_spec_to_mel(V):
    #尝试从谱图得到mfcc,和mfcc预测有关
    n_mfcc = 20
    dct_type = 2
    norm = 'ortho'
    v=V
    a,b=np.shape(v)
    if a>b:
        v=v.T
        a,b=np.shape(v)
    D=v**2
    S = librosa.feature.melspectrogram(S=D)
    S = librosa.power_to_db(S)

    return dct(S, axis=0, type=dct_type, norm=norm)[:n_mfcc]


def ming_zi_zhen_nan_qu(V1,V2,stft,res):
    n=np.shape(V1)
    a,b = np.shape(stft)
    stft1 = np.zeros((a, b),dtype=np.complex128)
    stft2 = np.zeros((a, b),dtype=np.complex128)
    for i in range(n[0]):
        v1=V1[i]
        v2=V2[i]
        p1=v1/(v1+v2+1e-50)
        p2=v2/(v1+v2+1e-50)
        stft_tmp=stft[:,i*n[2]:(i+1)*n[2]]
        stft1[:, i * n[2]:(i + 1) * n[2]] = np.multiply(p1, stft_tmp)
        stft2[:, i * n[2]:(i + 1) * n[2]] = np.multiply(p2, stft_tmp)
    stft1=np.hstack((stft1,res))
    stft2 = np.hstack((stft2, res))
    return stft1,stft2



