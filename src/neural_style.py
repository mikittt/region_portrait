import numpy as np
import six
import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import Variable
import time

import util

class NeuralStyle(object):
    def __init__(self, model, optimizer, content_weight, style_weight, tv_weight, content_layers, style_layers, resolution_num=1, device_id=-1, initial_image='random', keep_color=False):
        self.model = model
        self.optimizer = optimizer
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.device_id = device_id
        self.content_layer_names = content_layers
        self.style_layer_names = style_layers
        self.resolution_num = resolution_num
        self.initial_image = initial_image
        self.keep_color = keep_color
        if device_id >= 0:
            self.xp = cuda.cupy
            self.model.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, content_image, style_image, epoch_num, callback=None):
        device_id = None
        if self.device_id >= 0:
            device_id = self.device_id
        with cuda.Device(device_id):
            return self.__fit(content_image, style_image, epoch_num, callback)

    def __fit(self, content_image, style_image, epoch_num, callback=None):
        xp = self.xp
        input_image = None
        height, width = content_image.shape[-2:]
        base_epoch = 0
        old_link = None
        for stlide in [4, 2, 1][-self.resolution_num:]:
            if width // stlide < 64:
                continue
            content_x = Variable(xp.asarray(content_image[:,:,::stlide,::stlide]), volatile=True)
            if self.keep_color:
                style_x = Variable(util.luminance_only(xp.asarray(style_image[:,:,::stlide,::stlide]), content_x.data), volatile=True)
            else:
                style_x = Variable(xp.asarray(style_image[:,:,::stlide,::stlide]), volatile=True)
            content_layer_names = self.content_layer_names
            content_layers = self.model(content_x)
            #content_layers = [(name, content_layers[name]) for name in content_layer_names]
            style_layer_names = self.style_layer_names
            style_layers = self.model(style_x)
            style_grams = [(name, util.gram_matrix(style_layers[name])) for name in style_layer_names]
            if input_image is None:
                if self.initial_image == 'content':
                    input_image = xp.asarray(content_image[:,:,::stlide,::stlide])
                else:
                    input_image = xp.random.normal(0, 1, size=content_x.data.shape).astype(np.float32) * 0.001
            else:
                input_image = input_image.repeat(2, 2).repeat(2, 3)
                h, w = content_x.data.shape[-2:]
                input_image = input_image[:,:,:h,:w]
            link = chainer.Link(x=input_image.shape)
            if self.device_id >= 0:
                link.to_gpu()
            link.x.data[:] = xp.asarray(input_image)
            self.optimizer.setup(link)
            for epoch in six.moves.range(epoch_num):
                loss_info = self.__fit_one(link, content_layers, style_grams)
                if callback:
                    callback(base_epoch + epoch, link.x, loss_info)
            base_epoch += epoch_num
            input_image = link.x.data
        return link.x

    def __fit_one(self, link, content_layers, style_grams):
        xp = self.xp
        link.zerograds()
        layers = self.model(link.x)
        if self.keep_color:
            trans_layers = self.model(util.gray(link.x))
        else:
            trans_layers = layers
        loss_info = []
        loss = Variable(xp.zeros((), dtype=np.float32))
        for name, content_layer in content_layers:
            layer = layers[name]
            content_loss = self.content_weight * F.mean_squared_error(layer, Variable(content_layer.data))
            loss_info.append(('content_' + name, float(content_loss.data)))
            loss += content_loss
        for name, style_gram in style_grams:
            gram = util.gram_matrix(trans_layers[name])
            style_loss = self.style_weight * F.mean_squared_error(gram, Variable(style_gram.data))
            loss_info.append(('style_' + name, float(style_loss.data)))
            loss += style_loss
        tv_loss = self.tv_weight * util.total_variation(link.x)
        loss_info.append(('tv', float(tv_loss.data)))
        loss += tv_loss
        loss.backward()
        self.optimizer.update()
        return loss_info

class MRF(object):
    def __init__(self, model, optimizer, content_weight, style_weight, tv_weight, content_layers, style_layers, resolution_num=1, device_id=-1, initial_image='content', keep_color=False):
        self.model = model
        self.optimizer = optimizer
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        self.device_id = device_id
        self.content_layer_names = content_layers
        self.style_layer_names = style_layers
        self.resolution_num = resolution_num
        self.initial_image = initial_image
        self.keep_color = keep_color
        self.start_time = time.time()
        if device_id >= 0:
            self.xp = cuda.cupy
            self.model.to_gpu(device_id)
        else:
            self.xp = np

    def fit(self, content_image, style_image, content_mask, style_mask,  epoch_num, callback=None):
        device_id = None
        if self.device_id >= 0:
            device_id = self.device_id
        with cuda.Device(device_id):
            return self.__fit(content_image, style_image, content_mask, style_mask, epoch_num, callback)

    def __fit(self, content_image, style_image, input_content_mask, input_style_mask, epoch_num, callback=None):
        xp = self.xp
        input_image = None
        height, width = content_image.shape[-2:]
        base_epoch = 0
        mask_num = input_content_mask.shape[1]
        for stlide in [4, 2, 1][-self.resolution_num:]:
            if width // stlide < 64:
                continue
            content_x = Variable(xp.asarray(content_image[:,:,::stlide,::stlide]), volatile=True)
            content_mask = Variable(xp.asarray(input_content_mask[:,:,::stlide,::stlide],dtype=xp.float32),volatile=True)
            style_mask = Variable(xp.asarray(input_style_mask[:,:,::stlide,::stlide],dtype=xp.float32), volatile=True)
            
            
            if self.keep_color:
                style_x = Variable(util.luminance_only(xp.asarray(style_image[:,:,::stlide,::stlide]), content_x.data), volatile=True)
            else:
                style_x = Variable(xp.asarray(style_image[:,:,::stlide,::stlide]), volatile=True)
            
            #content_layer_names = self.content_layer_names
            _,content_layer4_2 = self.model(content_x)
            #content_layers = [(name, content_layers[name]) for name in content_layer_names]
            #style_layer_names = self.style_layer_names
            

            style_layer3_2,style_layer4_2 = self.model(style_x)

            content3_2_mask = F.max_pooling_2d(F.max_pooling_2d(content_mask,2,stride=2),2,stride=2)
            style3_2_mask = F.max_pooling_2d(F.max_pooling_2d(style_mask,2,stride=2),2,stride=2)
            content4_2_mask = F.max_pooling_2d(content3_2_mask,2,stride=2)
            style4_2_mask = F.max_pooling_2d(style3_2_mask,2,stride=2)
            """
            patch3_2 = []
            norm3_2 = []
            patch4_2 = []
            norm4_2 = []
            
            for i in range(mask_num):
                tmp_patch3_2, tmp_norm3_2 = util.patch(style3_2_mask[0,i,:,:].data[xp.newaxis,xp.newaxis,:,:]*style_layer3_2)
                tmp_patch4_2, tmp_norm4_2 = util.patch(style4_2_mask[0,i,:,:].data[xp.newaxis,xp.newaxis,:,:]*style_layer4_2)
                patch3_2.append(cuda.to_cpu(tmp_patch3_2))
                norm3_2.append(cuda.to_cpu(tmp_norm3_2))
                patch4_2.append(cuda.to_cpu(tmp_patch4_2))
                norm4_2.append(cuda.to_cpu(tmp_norm4_2))
            print(1)
            np.save("data/style_3_2_0_"+str(stlide)+".npy",np.array(patch3_2))
            np.save("data/style_3_2_1_"+str(stlide)+".npy",np.array(norm3_2))
            np.save("data/style_4_2_0_"+str(stlide)+".npy",np.array(patch4_2))
            np.save("data/style_4_2_1_"+str(stlide)+".npy",np.array(norm4_2)) 
            """
            print(2)
        
            style_patch3_2=(xp.array(np.load("data/style_3_2_0_"+str(stlide)+".npy")),xp.array(np.load("data/style_3_2_1_"+str(stlide)+".npy")))
            style_patch4_2=(xp.array(np.load("data/style_4_2_0_"+str(stlide)+".npy")),xp.array(np.load("data/style_4_2_1_"+str(stlide)+".npy")))

            print(3)
            if input_image is None:
                if self.initial_image == 'content':
                    input_image = xp.asarray(content_image[:,:,::stlide,::stlide])
                else:
                    input_image = xp.random.uniform(-20, 20, size=content_x.data.shape).astype(np.float32)
            else:
                input_image = input_image.repeat(2, 2).repeat(2, 3)
                h, w = content_x.data.shape[-2:]
                input_image = input_image[:,:,:h,:w]
            link = chainer.Link(x=input_image.shape)
            if self.device_id >= 0:
                link.to_gpu()
            link.x.data[:] = xp.asarray(input_image)
            self.optimizer.setup(link)
            for epoch in six.moves.range(epoch_num):
                loss_info = self.__fit_one(link, content_layer4_2.data, style_patch3_2, style_patch4_2, content3_2_mask.data, content4_2_mask.data)
                if callback:
                    callback(base_epoch + epoch, link.x, loss_info)
            base_epoch += epoch_num
            input_image = link.x.data
            print time.time()-self.start_time,"s"
        return link.x

    def __fit_one(self, link, content4_2, style3_2, style4_2, content3_2_mask, content4_2_mask):
        xp = self.xp
        mask_num = content3_2_mask.shape[1]
        link.zerograds()
        layer3_2,layer4_2 = self.model(link.x)
        if self.keep_color:
            #trans_layers = self.model(util.gray(link.x))
            print "don't keep color!"
        loss_info = []
        loss = Variable(xp.zeros((), dtype=np.float32))
        #layer = layers[name]
        content_loss = self.content_weight * F.mean_squared_error(layer4_2, Variable(content4_2))
        loss_info.append(('content_', float(content_loss.data)))
        loss += content_loss

        style_patch, style_patch_norm =  style3_2
        for i in range(mask_num):
            near,size,size2 = util.nearest_neighbor_patch(content3_2_mask[0,i,:,:][xp.newaxis,xp.newaxis,:,:]*layer3_2, style_patch[i], style_patch_norm[i])
            style_loss = F.sum(F.square(content3_2_mask[0,i,:,:][xp.newaxis,xp.newaxis,:,:]*layer3_2))*(self.style_weight *size2/size)-F.sum(near)*(2.*self.style_weight/size) 
            loss_info.append(('style_', float(style_loss.data)))
            loss+=style_loss
        
        style_patch, style_patch_norm =  style4_2
        for i in range(mask_num):
            near,size,size2 = util.nearest_neighbor_patch(content4_2_mask[0,i,:,:][xp.newaxis,xp.newaxis,:,:]*layer4_2, style_patch[i], style_patch_norm[i])
            style_loss =  F.sum(F.square(content4_2_mask[0,i,:,:][xp.newaxis,xp.newaxis,:,:]*layer4_2))*(self.style_weight *size2/size)-F.sum(near)*(2.*self.style_weight/size) 
            loss_info.append(('style_', float(style_loss.data)))
            loss+= style_loss
        
        tv_loss = self.tv_weight * util.total_variation(link.x)
        loss_info.append(('tv', float(tv_loss.data)))
        #loss += tv_loss
        loss.backward()
        self.optimizer.update()
        return loss_info
