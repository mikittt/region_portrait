import numpy as np
import six

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import Variable

def total_variation(x):
    return F.mean_squared_error(x[:,:,:-1,:],x[:,:,1:,:])+F.mean_squared_error(x[:,:,:,:-1],x[:,:,:,1:])

def gram_matrix(x):
    b, ch, h, w = x.data.shape
    v = F.reshape(x, (b, ch, w * h))
    return F.batch_matmul(v, v, transb=True) / np.float32(ch * w * h)

def patch(x, ksize=3, stride=1, pad=0):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    patch = xp.array([x.data[0,:,i:i+ksize,j:j+ksize].tolist() for i in range(h-2) for j in range(w-2)],dtype=xp.float32)
    print "ok"
    patch_norm = xp.linalg.norm(patch.reshape(patch.shape[0],-1),axis=1)[xp.newaxis,:,xp.newaxis,xp.newaxis]
    return patch,patch_norm

def gray(x):
    xp = cuda.get_array_module(x.data)
    w = Variable(xp.asarray([[[[0.114]], [[0.587]], [[0.299]]], [[[0.114]], [[0.587]], [[0.299]]], [[[0.114]], [[0.587]], [[0.299]]]], dtype=np.float32), volatile=x.volatile)
    return F.convolution_2d(x, W=w)

def nearest_neighbor_patch(x, patch, patch_norm):
    xp = cuda.get_array_module(x.data)
    conv = F.convolution_2d(x, W=patch, stride=1, pad=0)
    normal = conv.data/patch_norm
    size = normal.shape[2]*normal.shape[3]
    min_index = xp.argmax(normal,axis=1).reshape(-1)
    index_3 = np.arange(size)%normal.shape[3]
    index_2 = np.arange(size)/normal.shape[3]
    near = conv[0,min_index.tolist(),index_2.tolist(),index_3.tolist()]
    return near,float(normal.shape[2]*normal.shape[3]*patch[0].size),patch[0,0,:,:].size

def luminance_only(x, y):
    xp = cuda.get_array_module(x)
    w = xp.asarray([0.114, 0.587, 0.299], dtype=np.float32)
    x_shape = x.shape
    y_shape = y.shape

    x = x.reshape(x_shape[:2] + (-1,))
    xl = xp.zeros((x.shape[0], 1, x.shape[2]), dtype=np.float32)
    for i in six.moves.range(len(x)):
        xl[i,:] = w.dot(x[i])
    xl_mean = xp.mean(xl, axis=2, keepdims=True)
    xl_std = xp.std(xl, axis=2, keepdims=True)

    y = y.reshape(y_shape[:2] + (-1,))
    yl = xp.zeros((y.shape[0], 1, y.shape[2]), dtype=np.float32)
    for i in six.moves.range(len(y)):
        yl[i,:] = w.dot(y[i])
    yl_mean = xp.mean(yl, axis=2, keepdims=True)
    yl_std = xp.std(yl, axis=2, keepdims=True)

    xl = (xl - xl_mean) / xl_std * yl_std + yl_mean
    return xp.repeat(xl, 3, axis=1).reshape(x_shape)

def bgr_to_yiq(x):
    transform = np.asarray([[0.114, 0.587, 0.299], [-0.322, -0.274, 0.596], [0.312, -0.523, 0.211]], dtype=np.float32)
    n, c, h, w = x.shape
    x = x.transpose((1, 0, 2, 3)).reshape((c, -1))
    x = transform.dot(x)
    return x.reshape((c, n, h, w)).transpose((1, 0, 2, 3))

def yiq_to_bgr(x):
    transform = np.asarray([[1, -1.106, 1.703], [1, -0.272, -0.647], [1, 0.956, 0.621]], dtype=np.float32)
    n, c, h, w = x.shape
    x = x.transpose((1, 0, 2, 3)).reshape((c, -1))
    x = transform.dot(x)
    return x.reshape((c, n, h, w)).transpose((1, 0, 2, 3))

def split_bgr_to_yiq(x):
    x = bgr_to_yiq(x)
    y = x[:,0:1,:,:]
    iq = x[:,1:,:,:]
    return np.repeat(y, 3, axis=1), iq

def join_yiq_to_bgr(y, iq):
    y = bgr_to_yiq(y)[:,0:1,:,:]
    return yiq_to_bgr(np.concatenate((y, iq), axis=1))

def match_color_histogram(x, y):
    z = np.zeros_like(x)
    shape = x[0].shape
    for i in six.moves.range(len(x)):
        a = x[i].reshape((3, -1))
        a_mean = np.mean(a, axis=1, keepdims=True)
        a_var = np.cov(a)
        d, v = np.linalg.eig(a_var)
        a_sigma_inv = v.dot(np.diag(d ** (-0.5))).dot(v.T)

        b = y[i].reshape((3, -1))
        b_mean = np.mean(b, axis=1, keepdims=True)
        b_var = np.cov(b)
        d, v = np.linalg.eig(b_var)
        b_sigma = v.dot(np.diag(d ** 0.5)).dot(v.T)

        transform = b_sigma.dot(a_sigma_inv)
        z[i,:] = (transform.dot(a - a_mean) + b_mean).reshape(shape)
    return z
