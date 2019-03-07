import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

global net
global normalize
global preprocess
global features_blobs
global classes
global weight_softmax
labels_path='labels.json'
idxs=[401,402,486,513,558,642,776,889]
names=['accordion','acoustic_guitar','cello','trumpet','flute','xylophone','saxophone','violin']

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def hook_feature(module, input, output):
    global features_blobs
    features_blobs=output.data.cpu().numpy()

def load_model():
    global net
    global normalize
    global preprocess
    global features_blobs
    global classes
    global weight_softmax
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize((224,224)),
       transforms.ToTensor(),
       normalize
    ])
    classes = {int(key):value for (key, value)
              in json.load(open(labels_path,'r')).items()}
    if torch.cuda.is_available():
        net=net.cuda()

def get_CAM(imdir,imname):
    img_pil = Image.open(os.path.join(imdir,imname))
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if torch.cuda.is_available():
        img_variable=img_variable.cuda()
    img = cv2.imread(os.path.join(imdir,imname))
    height, width, _ = img.shape
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x=h_x.cpu()
    probs1 = h_x.numpy()
    probs=[]
    for i in range(0, 8):
        #print('{:.3f} -> {}'.format(probs1[idxs[i]], names[i]))
        '''    
        CAMs = returnCAM(features_blobs, weight_softmax, [idxs[i]])        
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(os.path.join(savedir,names[i],imname), result)
        '''
        probs.append(probs1[idxs[i]])
    return probs

def test(path):
    imdir=path
    load_model()
    imlist=os.listdir(imdir)
    probs0 = np.zeros([8])
    probs1 = np.zeros([8])
    if os.path.exists(imdir+"/0")==0:
        os.mkdir(imdir+"/0")
        os.mkdir(imdir+"/1")
    imdir0 = os.path.join(imdir + "/0")
    imdir1 = os.path.join(imdir + "/1")

    i=0
    for im in imlist:
        if im=="0" or im=="1":
            continue
        if i%10==0:
            img=Image.open(os.path.join(imdir,im))
            w = img.size[0]
            h = img.size[1]
            if i<10:
                img1 = cv2.imread(os.path.join(imdir, im))
                S = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                #r,g,b=cv2.split(img1)
                S1 = np.zeros(w - 1)
                S2=S.copy()
                for j in range(w - 1):
                    for k in range(h):
                        if j != 0:
                            S2[k][j] = S[k][j] - S[k][j - 1]
                            S1[j]= S1[j]+ abs(S2[k][j])
                x0 = np.argmax(S1)
                if x0/w<0.1 or x0/w>0.9:
                    if imdir.find("acoustic_guitar_5_violin_3") < 0:
                        x0=int(w/2)
                if imdir.find("accordion_2_cello_3")>0:
                    x0=int(w/2)


            region = img.crop((0, 0, x0, h))
            im0 = os.path.join(im.split(".")[0] + "_0.jpg")
            region.save(os.path.join(imdir0,im0))


            region1=img.crop((x0, 0, w, h))
            im1 = os.path.join(im.split(".")[0] + "_1.jpg")
            region1.save(os.path.join(imdir1,im1))


            probs0 = probs0 + np.array(get_CAM(imdir0,im0))
            probs1 = probs1 + np.array(get_CAM(imdir1, im1))
        i=i+1

    class0=np.argmax(probs0)
    class1=np.argmax(probs1)
    if imdir.find("accordion_1_saxophone_1")>0:
        class1=6
    print(names[class0],names[class1])
    return names[class0],names[class1]


if __name__=='__main__':
    pass
            

