import os
import tensorflow as tf
import numpy as np
from utils import check_folder, dataGenerator, montage, load_mnist, load_anime
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageio import imsave

slim = tf.contrib.slim

class CGAN(object):
    model_name = "CGAN"
    
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
                       
            #the size of the first layer of generator
            self.s_size = 3
            #arguments for the last layer of generator
            self.last_dconv = {
                    'kernel_size': [5,5],
                    'stride': 1,
                    'padding': 'VALID'}
            #depths for convolution in generator and discriminator
            self.g_depths = [512, 256, 128, 64]
            self.d_depths = [64, 128, 256, 512]
            
            #channel and label dim
            self.c_dim = 1
            self.y_dim = 10
            
            #WGAN parameter, the number of critic iterations for each epoch
            self.d_iters = 1
            self.g_iters = 2
            
            #train
            self.learning_rate = 0.0002
            self.beta1 = 0.5
            
            #test, number of generated images to be saved 
            self.sample_num = 100
            
            #load numpy array of images and labels
            self.images, self.labels = load_mnist(self.dataset_name)
            
        elif dataset_name == 'anime':        
            #image_dimension 
            self.imgH = 64
            self.imgW = 64
            
            #the size of the first layer of generator
            self.s_size = 4
            #arguments for the last layer of generator, same as the general 
            self.last_dconv = {}
            
            #depths for convolution in generator and discriminator
            self.g_depths = [512, 256, 128, 64]
            self.d_depths = [64, 128, 256, 512]
            
            #channel and label dim
            self.c_dim = 3
            self.y_dim = 22
            
            #WGAN parameter, the number of critic iterations for each epoch
            self.d_iters = 1
            self.g_iters = 4
            
            #train
            self.learning_rate = 0.0002
            self.beta1 = 0.5
            self.beta2 = 0.9
            
            #test, number of generated images to be saved 
            self.sample_num = 64
            
            self.images, self.labels = load_anime(self.dataset_name)
            
        else:
            raise NotImplementedError
    
    def batch_norm_params(self, is_training):
        return {
                'is_training': is_training,
                'decay': 0.9,
                'zero_debias_moving_mean': True,
                'epsilon': 1e-5}
    
    def gen_arg_scope(self, is_training=True):
        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=self.batch_norm_params(is_training)):
            with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=[4,4], 
                            stride=2,
                            padding='SAME') as arg_scp:
                return arg_scp
    
    
    def disc_arg_scope(self, is_training=True):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.leaky_relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=self.batch_norm_params(is_training)):
            with slim.arg_scope([slim.conv2d],
                                kernel_size=[4,4],
                                stride=2,
                                padding='SAME') as arg_scp:
                return arg_scp
    
    def discriminator(self, x, y, is_training=True, reuse=False):
        with tf.variable_scope('discriminator', values=[x], reuse=reuse): 
            with slim.arg_scope(self.disc_arg_scope(is_training)):
                net = tf.concat([x, y], axis = 3)
                
                for i in range(len(self.d_depths)):
                    scp = 'conv%i' % (i+1)
                    net = slim.conv2d(net, self.d_depths[i], scope=scp)
                
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1, 
                                           activation_fn=None,
                                           normalizer_fn=None,
                                           normalizer_params=None,
                                           scope='real_or_fake')
                return net
    
    def generator(self, z, y, is_training=True, reuse=False):
        with tf.variable_scope('generator', values=[z], reuse=reuse):
            with slim.arg_scope(self.gen_arg_scope(is_training)):
                
                z = tf.concat([z, y], axis = 1)
                
                net = slim.fully_connected(z, self.s_size*self.s_size*self.g_depths[0], 
                                           normalizer_fn=None, 
                                           normalizer_params=None, 
                                           scope = 'projection')
                net = tf.reshape(net, [-1, self.s_size, self.s_size, self.g_depths[0]])
                net = slim.batch_norm(net, scope='batch_norm', **self.batch_norm_params(is_training))
                
                for i in range(1, len(self.g_depths)):
                    scp = 'deconv%i' % i
                    net = slim.conv2d_transpose(net, self.g_depths[i], scope=scp)
                
                #last layer has different normalizer and activation
                net = slim.conv2d_transpose(net, self.c_dim,
                                        activation_fn=tf.nn.tanh,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope='deconv%i' % len(self.g_depths),
                                        **self.last_dconv)
            
                return net
    
    def build_model(self):        
        """Graph input"""
        #images
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.imgH, self.imgW, self.c_dim], name = 'real_images')
        if self.dataset_name == 'mnist' or self.dataset_name == 'fashion-mnist':
            self.rotated_image = self.inputs
        elif self.dataset_name == 'anime':
            images_fliped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), self.inputs)
            angles = tf.random_uniform([self.batch_size], minval = -15.0 * np.pi / 180, 
                                       maxval= 15.0 * np.pi / 180, dtype = tf.float32)
            self.rotated_image = tf.contrib.image.rotate(images_fliped, angles, interpolation='NEAREST')
        
        #labels for generator and discriminator individually
        self.y_gen = tf.placeholder(dtype=tf.float32, shape=[None, self.y_dim], name = 'label_gen')
        self.y_dis = tf.placeholder(dtype=tf.float32, shape=[None, self.imgH, self.imgW, self.y_dim], name = 'label_dis')
        
        #noise
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim], name='noise')
        
        """Graph output"""
        #output of D for fake images
        G = self.generator(self.z, self.y_gen, is_training=True, reuse=tf.AUTO_REUSE)
        D_fake = self.discriminator(G, self.y_dis, is_training=True, reuse=tf.AUTO_REUSE)
        #output of D for real images
        D_real = self.discriminator(self.rotated_image, self.y_dis, is_training=True, reuse=tf.AUTO_REUSE)
        
        #get loss for discriminator
        d_loss_real = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(logits=D_real, 
                                                                       multi_class_labels=tf.ones_like(D_real)))
        
        d_loss_fake = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(logits=D_fake,
                                                                       multi_class_labels=tf.zeros_like(D_fake)))
        self.d_loss = d_loss_real + d_loss_fake
    
        self.g_loss = tf.reduce_mean(slim.losses.sigmoid_cross_entropy(logits=D_fake,
                                                                       multi_class_labels=tf.ones_like(D_fake)))
        
        """Training"""
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        
        #optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)
        
        self.d_loss_real_sum = tf.summary.scalar('d_loss_real', d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar('d_loss_fake', d_loss_fake)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)
        
        self.d_sum = tf.summary.merge([self.d_loss_real_sum, self.d_loss_fake_sum, self.d_loss_sum])
        self.g_sum = tf.summary.scalar('g_loss', self.g_loss)
        
        """Testing"""
        self.predict_images = self.generator(self.z, self.y_gen, is_training=False, reuse=tf.AUTO_REUSE)
    
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
                batch_dis = np.copy(batch_labels)
                batch_dis = np.reshape(batch_dis, [-1, 1, 1, self.y_dim])
                batch_dis = batch_dis * np.ones([self.batch_size, self.imgH, self.imgW, self.y_dim])
                _, d_loss, sumStr = self.sess.run([self.d_optimizer, self.d_loss, self.d_sum], 
                                                  feed_dict={self.inputs: batch_images, self.y_gen: batch_labels, self.z:batch_z, self.y_dis:batch_dis})
            self.writer.add_summary(sumStr, epoch)

            for _ in range(self.g_iters):
                batch_z = np.random.uniform(low = -1., high = 1., size = (self.batch_size, self.z_dim))
                _, g_loss, sumStr = self.sess.run([self.g_optimizer, self.g_loss, self.g_sum], 
                                                  feed_dict={self.inputs: batch_images, self.z:batch_z, self.y_gen: batch_labels, self.y_dis: batch_dis})
            self.writer.add_summary(sumStr, epoch)
            
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
        samples = self.sess.run(self.predict_images, feed_dict={self.z: z_sample, self.y_gen: y_sample})
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
                
            samples = self.sess.run(self.predict_images, feed_dict={self.z: z_sample, self.y_gen: y_labels})
            samples = montage(samples)
            image_path = check_folder(self.sample_dir + '/' + self.model_dir) + '/' + 'sample.jpg'   
            imsave(image_path, samples)
            print("[*] Load Successfully")
        else:
            print("[!] Load failed...")

        
