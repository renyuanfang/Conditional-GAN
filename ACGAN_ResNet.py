import os
import tensorflow as tf
import numpy as np
from utils import check_folder, dataGenerator, montage, load_mnist, load_anime
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageio import imsave
from ops import lrelu, conv2d,batch_norm

slim = tf.contrib.slim

class ACGAN_ResNet(object):
    model_name = "ACGAN_ResNet"
    
    def __init__(self, sess, epoch, batch_size,
                 z_dim, dataset_name, checkpoint_dir,
                 sample_dir, log_dir, mode):
        self.sess = sess
        self.epoch = epoch
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.log_dir = log_dir
        self.dataset_name = dataset_name
        self.z_dim = z_dim
        self.random_seed = 1000
        
        if dataset_name == 'mnist' or dataset_name == 'fashion-mnist':
            #image_dimension 
            self.imgH = 28
            self.imgW = 28
            
            #feature map size of the first layer of generator
            self.s_size = 7
            self.g_depth = 64 #first layer
            self.d_depths = [64, 128, 256, 512]
            self.num_layers = 2 #how many times upsampling
            
            #channel and label dim
            self.c_dim = 1
            self.y_dim = 10
            
            self.d_iters = 2
            self.g_iters = 1
            
            self.learning_rate = 0.0002
            self.beta1 = 0.5
            self.beta2 = 0.999
            
            self.LAMBDA = 5
            self.BETA = 3
            
            #test, number of generated images to be saved 
            self.sample_num = 100
            
            #load numpy array of images and labels
            self.images, self.labels = load_mnist(self.dataset_name)
            
        elif dataset_name == 'anime':        
            #image_dimension
            self.imgH = 64
            self.imgW = 64
            
            #feature map size of the first layer of generator
            self.s_size = 16
            self.g_depth = 64 #first layer
            self.d_depths = [64, 128, 256, 512]
            self.num_layers = 2 #how many times upsampling
            
            #channel and label dim
            self.c_dim = 3
            self.y_dim = 22
            
            self.d_iters = 2
            self.g_iters = 1
            
            self.learning_rate = 0.0002
            self.beta1 = 0.5
            self.beta2 = 0.999
            
            self.LAMBDA = 0.05
            self.BETA = 10
            
            #test, number of generated images to be saved 
            self.sample_num = 64
            
            self.images, self.labels = load_anime(self.dataset_name)
            
        else:
            raise NotImplementedError
    
    def d_block(self, inputs, filters):
        h = lrelu(conv2d(inputs, filters, 3, 1))
        h = conv2d(h, filters, 3, 1)
        h = lrelu(tf.add(h, inputs))
        return h
    
    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator', values=[x], reuse=reuse): 
            net = x
            
            for filters in self.d_depths:
                net = lrelu(conv2d(net, filters, 3, 2)) 
                net = self.d_block(net, filters)
                net = self.d_block(net, filters)
            
            net = tf.contrib.layers.flatten(net)
            real_fake = tf.layers.dense(net, units=1)
            y_ = tf.layers.dense(net, units=self.y_dim)
            return real_fake, y_
    
    def g_block(self, inputs, is_training):
        h = tf.nn.relu(batch_norm(conv2d(inputs, self.g_depth, 3, 1, use_bias=False), is_training))
        h = batch_norm(conv2d(h, self.g_depth, 3, 1, use_bias=False), is_training)
        h = tf.add(inputs, h)
        return h
    
    def generator(self, z, y, is_training=True, reuse=False):
        with tf.variable_scope('generator', values=[z], reuse=reuse):
            z = tf.concat([z, y], axis = 1)
            
            net = tf.layers.dense(z, units = self.s_size * self.s_size * self.g_depth)
            net = tf.reshape(net, [-1, self.s_size, self.s_size, self.g_depth])
            net = tf.nn.relu(batch_norm(net, is_training))
            shortcut = net
            
            #define 16 res block, do not change the feature map size and number
            for i in range(16):
                net = self.g_block(net, is_training)
            
            net = tf.nn.relu(batch_norm(net, is_training))
            net = tf.add(net, shortcut)
            
            #sub-pixel, convolution, upsampling
            for i in range(self.num_layers):
                net = conv2d(net, self.g_depth * 4, 3, 1, use_bias=False)
                net = tf.depth_to_space(net, 2) #change from imgH * imgW * 256 to 2*imgH * 2*imgW * 64
                net = tf.nn.relu(batch_norm(net, is_training))
            
            net = tf.layers.conv2d(net, filters=self.c_dim, kernel_size=5, 
                                   strides=1, padding='SAME', 
                                   activation=tf.nn.tanh, use_bias=True)
            
            return net   
    
    def get_random_labels(self, num):
        labels = np.zeros([num, self.y_dim])
        if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
            index = np.random.randint(low = 0, high = self.y_dim, size = [num])
            labels[np.arange(num), index] = 1
        elif self.dataset_name == 'anime':
            hair_label = np.random.randint(0, 12, num)
            eye_label = np.random.randint(0, 10, num)
            labels[np.arange(num), hair_label] = 1
            labels[np.arange(num), 12 + eye_label] = 1
        
        return labels    
    
    def build_model(self):        
        """Graph input"""
        #images
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.imgH, 
                                                              self.imgW, self.c_dim], name='real_image')
        if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
            self.aug_inputs = self.inputs
        elif self.dataset_name == 'anime':
            self.aug_inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), self.inputs)
            
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.y_dim], name='label')
        self.y_noise = tf.placeholder(dtype=tf.float32, shape=[None, self.y_dim], name = 'noise_label')
        
        #noise
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='noise')
        self.input_perturb = tf.placeholder(dtype=tf.float32, shape = [None, self.imgH, self.imgW, 
                                                                       self.c_dim], name='real_image_perturb')
        
        """Graph output"""
        #output of D for fake images
        G = self.generator(self.z, self.y_noise, is_training=True, reuse=tf.AUTO_REUSE)
        D_fake, y_fake = self.discriminator(G, reuse=tf.AUTO_REUSE)
        D_real, y_real = self.discriminator(self.aug_inputs, reuse=tf.AUTO_REUSE)
        
        #get loss for discriminator
        d_loss_real = slim.losses.sigmoid_cross_entropy(logits=D_real, 
                                                        multi_class_labels=tf.ones_like(D_real))        
        d_loss_fake = slim.losses.sigmoid_cross_entropy(logits=D_fake,
                                                        multi_class_labels=tf.zeros_like(D_fake))        
        d_class = slim.losses.sigmoid_cross_entropy(logits=y_real,
                                                    multi_class_labels=self.y)
        self.d_loss = d_loss_real + d_loss_fake + self.BETA * d_class
        
        #gradient penlty
        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = alpha * self.aug_inputs + (1 - alpha) * self.input_perturb
        D_inter= self.discriminator(interpolates, reuse=tf.AUTO_REUSE)[0]
        grad = tf.gradients(D_inter, [interpolates])[0]
        slope = tf.sqrt(tf.reduce_sum(tf.square(grad), axis = [1, 2, 3]))
        gp = tf.reduce_mean((slope - 1.) ** 2)
        self.d_loss += self.LAMBDA*gp
    
        g_class = slim.losses.sigmoid_cross_entropy(logits=y_fake, multi_class_labels=self.y_noise)
        self.g_loss = -tf.reduce_mean(D_fake) + self.BETA * g_class
        
        #use learning rate decay
        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.lr_decay = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.global_step,
                                              decay_steps=20000, decay_rate=0.5)
        
        """Training"""
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')

        #optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_decay, beta1 = self.beta1, beta2 = self.beta2).minimize(self.d_loss, var_list=d_vars)
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr_decay, beta1 = self.beta1, beta2 = self.beta2).minimize(self.g_loss, var_list=g_vars)
        
        self.d_loss_real_sum = tf.summary.scalar('d_loss_real', d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', d_loss_fake)
        self.d_class_sum = tf.summary.scalar('d_class', d_class)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum, self.d_class_sum])
        
        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
        self.g_class_sum = tf.summary.scalar('g_class', g_class)
        self.g_sum = tf.summary.merge([self.g_loss_sum, self.g_class_sum])
        
        """Testing"""
        self.predict_images = self.generator(self.z, self.y_noise, is_training=False, reuse=tf.AUTO_REUSE)
    
    def train(self):        
        #initialize all variables
        tf.global_variables_initializer().run()
        
        #used for visualization during the training, the noise is fixed
        np.random.seed(self.random_seed)
        self.sample_z = np.random.uniform(low = -1., high = 1., size = (self.sample_num, self.z_dim))
        if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
            self.sample_y = np.zeros((self.sample_num, self.y_dim))
            for i in range(self.y_dim):
                self.sample_y[i*self.y_dim:(i+1)*self.y_dim, i] = 1
        elif self.dataset_name == 'anime':
            self.sample_y = np.zeros((self.sample_num, self.y_dim))
            #blue hair, blue eye; blue hair, green eye; green hair, blue eye; green hair, red eye; pink hair, aqua eye; pink hair, purple eye; red hair, blue eye; red hair, brown eye 
            indices = [[8, 21], [8, 18], [4, 21], [4, 20], [7, 16], [7, 17], [5, 21], [5, 19]]
            for i in range(8):
                self.sample_y[i*8 : (i+1)*8, indices[i]] = 1 
        np.random.seed()
        
        #saver to save the model
        self.saver = tf.train.Saver(max_to_keep=2)
        
        #summary writer
        self.writer = tf.summary.FileWriter(check_folder(self.log_dir + '/' + self.model_dir), self.sess.graph)
        
        #restore check-point if it exists
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter
            print("[*] Load Successfully")
        else:
            start_epoch = 0
            print("[!] Load failed...")
            
        loss = {'d': [], 'g': []}       
        gen = dataGenerator(self.images, self.labels, self.dataset_name, self.batch_size, self.imgH, self.imgW)
        for epoch in tqdm(range(start_epoch, self.epoch)):
            for _ in range(self.d_iters):
                batch_z = np.random.uniform(low = -1., high = 1., size = (self.batch_size, self.z_dim))
                batch_images, batch_labels = next(gen)
                batch_images_perturb = batch_images + 0.5 * batch_images.std() * np.random.random(batch_images.shape)
                noise_label = self.get_random_labels(self.batch_size)
                _, d_loss, sumStr = self.sess.run([self.d_optimizer, self.d_loss, self.d_sum], feed_dict={self.inputs: batch_images, 
                                                  self.y: batch_labels, self.z:batch_z, self.y_noise: noise_label,
                                                  self.input_perturb: batch_images_perturb})
            self.writer.add_summary(sumStr, epoch)

            for _ in range(self.g_iters):
                batch_z = np.random.uniform(low = -1., high = 1., size = (self.batch_size, self.z_dim))
                noise_label = self.get_random_labels(self.batch_size)
                _, g_loss, sumStr = self.sess.run([self.g_optimizer, self.g_loss, self.g_sum], feed_dict={self.z:batch_z, self.y_noise: noise_label})
            self.writer.add_summary(sumStr, epoch)
            
            self.sess.run([self.add_global, self.lr_decay])
            
            loss['d'].append(d_loss)
            loss['g'].append(g_loss)
                
            if (epoch + 1) % 5000 == 0:
               print('Epoch: %d, d_loss: %.4f, g_loss: %.4f' % (epoch+1, d_loss, g_loss))
               #save model
               self.save(self.checkpoint_dir, epoch+1)
               #show temporal results
               self.visualize_results(epoch+1, self.sample_z, self.sample_y)

        plt.plot(loss['d'], label = 'Discirminator')
        plt.plot(loss['g'], label = 'Generator')
        plt.legend(loc = 'upper right')
        plt.savefig('Loss.png')
        plt.show()
        
    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)
    
    def save(self, checkpoint_dir, step):
        checkpoint_dir = check_folder(os.path.join(checkpoint_dir, self.model_dir))
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.dataset_name+".model"), global_step=step)
    
    def visualize_results(self, epoch, z_sample, y_sample):        
        samples = self.sess.run(self.predict_images, feed_dict={self.z: z_sample, self.y_noise: y_sample})
        samples = montage(samples)
        image_path = check_folder(self.sample_dir + '/' + self.model_dir) + '/' + 'epoch_%d' % epoch + '_test.jpg'   
        imsave(image_path, samples)
        
        #show the images
        plt.axis('off')
        if self.c_dim == 1:
            plt.imshow(samples, cmap = 'gray')
        else:
            plt.imshow(samples)
        plt.show()
        plt.close()
    
    def load(self, checkpoint_dir):
        import re
        print("[*] Reading checkpoints")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print("[*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print("[*] Failed to find a checkpoint")
            return False, 0
    
    def infer(self, y_labels=None):        
        tf.global_variables_initializer().run()
        
        #saver to save the model
        self.saver = tf.train.Saver(max_to_keep=2)
        
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:          
            z_sample = np.random.uniform(low = -1., high = 1., size = (self.sample_num, self.z_dim))
            if y_labels is None:
                if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
                    y_labels = np.zeros((self.sample_num, self.y_dim))
                    for i in range(self.y_dim):
                        y_labels[i*self.y_dim:(i+1)*self.y_dim, i] = 1
                elif self.dataset_name == 'anime':
                    y_labels = np.zeros((self.sample_num, self.y_dim))
                    #blue hair, blue eye; blue hair, green eye; green hair, blue eye; green hair, red eye; pink hair, aqua eye; pink hair, purple eye; red hair, blue eye; red hair, brown eye 
                    indices = [[8, 21], [8, 18], [4, 21], [4, 20], [7, 16], [7, 17], [5, 21], [5, 19]]
                    for i in range(8):
                        y_labels[i*8 : (i+1)*8, indices[i]] = 1
                
            samples = self.sess.run(self.predict_images, feed_dict={self.z: z_sample, self.y_noise: y_labels})
            samples = montage(samples)
            image_path = check_folder(self.sample_dir + '/' + self.model_dir) + '/' + 'sample.jpg'   
            imsave(image_path, samples)
            print("[*] Load Successfully")
        else:
            print("[!] Load failed...")



