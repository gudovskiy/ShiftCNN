import sys
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
#
N = 2
B = 4
#
#model = "squeezenet_v1.1"
model = "ResNet-50"
SOURCE_PATH = os.environ["HOME"]+"/github/caffe/models/"+model+"/"
prototxt = SOURCE_PATH+"train_val.prototxt"
source   = SOURCE_PATH+model+".caffemodel"
qtarget  = SOURCE_PATH+model+"_N"+str(N)+"_B"+str(B)+".caffemodel"
caffe_root = os.environ["CAFFE_ROOT"]
os.chdir(caffe_root)
print caffe_root
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()
net = caffe.Net(prototxt, source, caffe.TEST)
layers = net.params.keys()
linestyles = ['--', '-']

for idx, layer in enumerate(layers):
    #if not('bn' in layer) and not('scale' in layer): # do not include batch normalization and scaling layers in ResNets
        wT= 0.0
        w = net.params[layer][0].data
        wMax = np.max(np.abs(w))
        r = w/wMax # normalize
        for n in range(0, N):
            qSgn = np.sign(r)
            qLog = np.log2(abs(r+1e-32))
            qIdx = np.floor(qLog)
            bLog = qIdx + np.log2(1.5)
            bIdx = qLog > bLog # border condition
            qIdx[bIdx] = qIdx[bIdx] + 1.0
            q = qSgn * 2**(qIdx)
            qIdxMem = qSgn * (-(n+1)-qIdx+2)
            sIdx = (2-(n+1)-qIdx) > (2**(B-1)-1) # saturation condition
            q[sIdx] = 0
            qIdxMem[sIdx] = 0
            zIdx = q!=0
            wT += q
            r  -= q
        
        np.copyto(net.params[layer][0].data, wT*wMax)

net.save(qtarget)
