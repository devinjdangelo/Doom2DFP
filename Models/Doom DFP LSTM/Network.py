import tensorflow as tf
from tensorflow.contrib.slim import conv2d, fully_connected, flatten, dropout, conv2d_transpose
from tensorflow.python.keras.initializers import he_normal
from tensorflow.python.keras.layers import LeakyReLU


class DFP_Network():
    def __init__(self,a_size,num_offsets,num_measurements,xdim,ydim,using_pc,start_lr,half_lr_every_n_steps,main_weight,pc_weight):
        #Inputs and visual encoding layers
        #num_measurements[0] = num_observe_measurements
        #num_measurements[1] = num_predict_measuremnets
        
        self.xdim = xdim
        self.ydim = ydim
        self.num_measurements = num_measurements
        self.num_offsets = num_offsets
        self.a_size = a_size
        self.using_pc = using_pc
        
        self.start_lr = start_lr
        self.half_lr_every_n_steps = half_lr_every_n_steps
        self.main_weight = main_weight
        self.pc_weight = pc_weight
        
        self.outputdim = num_measurements[1]*num_offsets
        self.outputdim_a1 = num_measurements[1]*num_offsets*a_size[0]
        self.outputdim_a2 = num_measurements[1]*num_offsets*a_size[1]
        self.outputdim_a3 = num_measurements[1]*num_offsets*a_size[2]
        self.outputdim_a4 = num_measurements[1]*num_offsets*a_size[3]
        
        self.steps = tf.placeholder(shape=(),dtype=tf.int32)
        self._create_network()
        
        
    def _create_network(self):
        
        self._create_input_stream()
        self._create_LSTM()
        
        if self.using_pc:
            self._create_pc_net()
            
        self._create_base_net()
        self._create_loss_train_ops()
        
    def _create_input_stream(self):
        self.observation = tf.placeholder(shape=[None,self.xdim,self.ydim,3],dtype=tf.float32)
        self.conv1 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.observation,num_outputs=32,
            kernel_size=[8,8],stride=[4,4],padding='VALID')
        self.conv2 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv1,num_outputs=64,
            kernel_size=[4,4],stride=[2,2],padding='VALID')
        self.conv3 = conv2d(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.conv2,num_outputs=64,
            kernel_size=[3,3],stride=[1,1],padding='VALID')
        
        self.convout = fully_connected(flatten(self.conv3),512,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        
        self.measurements = tf.placeholder(shape=[None,self.num_measurements[0]],dtype=tf.float32)
        self.dense_m1 = fully_connected(flatten(self.measurements),128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        self.dense_m2 = fully_connected(self.dense_m1,128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        self.dense_m3 = fully_connected(self.dense_m2,128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        
        self.goals = tf.placeholder(shape=[None,self.num_measurements[1]],dtype=tf.float32)
        self.dense_g1 = fully_connected(flatten(self.goals),128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        self.dense_g2 = fully_connected(self.dense_g1,128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        self.dense_g3 = fully_connected(self.dense_g2,128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())  

        self.action_history = tf.placeholder(shape=[None,12],dtype=tf.float32)
        self.dense_a1 = fully_connected(flatten(self.goals),128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())

        
        self.merged_input1 = tf.concat([self.dense_a1,self.dense_m3,self.convout,self.dense_g3],1,name="InputMerge")

    
    def _create_LSTM(self): 
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(512)
        #default is testing or exploring, i.e. sequentially sess.run on one observation at a time, no batches
        #during train time, we will pass the dynamic sequence length we've chosen to train on for this set of batches.
        self.seq_length = tf.placeholder_with_default(1,shape=())
        self.c_in = tf.placeholder(shape=[None, self.lstm1.state_size.c],dtype=tf.float32)
        self.h_in = tf.placeholder(shape=[None, self.lstm1.state_size.h], dtype=tf.float32)
        rnn_in = tf.reshape(self.merged_input1, [-1,self.seq_length,896])
        state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
        
        self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
            self.lstm1, rnn_in, initial_state=state_in, time_major=False)
        
        self.lstm_outputs = tf.reshape(self.lstm_outputs, [-1, 512])
            
        self.exploring = tf.placeholder_with_default(False,shape=())
        
        dropoutrates = [0.9,0.6,0.40,0.30,0.2,0.15,0.1,.05,.01]
        boundaries = [500000,1000000,2000000,3500000,5000000,7500000,10000000,12500000]
        dropoutrate = tf.train.piecewise_constant(self.steps,boundaries,dropoutrates)
        
        #below tf op calculates -> keep_rate = 0.95 * 0.5 **(self.steps/1000000)
        #dropoutrate = tf.train.exponential_decay(.95,self.steps,1000000,0.5)

        self.merged_dropout = dropout(self.lstm_outputs,keep_prob=1-dropoutrate,is_training=self.exploring)

    def _create_pc_net(self):
        #dim of deconv xdim ->(xdim-1)*xstride + xkernal
        
        self.deconv_in_E = tf.reshape(self.merged_dropout,[-1,8,8,4])
        self.deconv_in_a1 = tf.tile(self.deconv_in_E,[1,1,1,self.a_size[0]])
        self.deconv_in_a2 = tf.tile(self.deconv_in_E,[1,1,1,self.a_size[1]])
        self.deconv_in_a3 = tf.tile(self.deconv_in_E,[1,1,1,self.a_size[2]])
        self.deconv_in_a4 = tf.tile(self.deconv_in_E,[1,1,1,self.a_size[3]])
        
        self.deconv_E = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_in_E,num_outputs=1,
            kernel_size=[6,6],stride=[2,2],padding='VALID')
        
        self.deconv_a1 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_in_a1,num_outputs=self.a_size[0],
            kernel_size=[6,6],stride=[2,2],padding='VALID')
            
        self.deconv_a2 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_in_a2,num_outputs=self.a_size[1],
            kernel_size=[6,6],stride=[2,2],padding='VALID')
            
        self.deconv_a3 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_in_a3,num_outputs=self.a_size[2],
            kernel_size=[6,6],stride=[2,2],padding='VALID')
            
        self.deconv_a4 = conv2d_transpose(activation_fn=LeakyReLU(0.2),
            weights_initializer=he_normal(),
            inputs=self.deconv_in_a4,num_outputs=self.a_size[3],
            kernel_size=[6,6],stride=[2,2],padding='VALID')
        
        self.deconv_a1 = self.deconv_a1 - tf.reduce_mean(self.deconv_a1,axis=3,keep_dims=True)
        self.deconv_a2 = self.deconv_a2 - tf.reduce_mean(self.deconv_a2,axis=3,keep_dims=True)
        self.deconv_a3 = self.deconv_a3 - tf.reduce_mean(self.deconv_a3,axis=3,keep_dims=True)
        self.deconv_a4 = self.deconv_a4 - tf.reduce_mean(self.deconv_a4,axis=3,keep_dims=True)
        
        self.deconv_a1_chosen = tf.placeholder(shape=[None,20,20,self.a_size[0]],dtype=tf.float32)
        self.deconv_a2_chosen = tf.placeholder(shape=[None,20,20,self.a_size[1]],dtype=tf.float32)
        self.deconv_a3_chosen = tf.placeholder(shape=[None,20,20,self.a_size[2]],dtype=tf.float32)
        self.deconv_a4_chosen = tf.placeholder(shape=[None,20,20,self.a_size[3]],dtype=tf.float32)
        
        self.deconv_a1_pred = tf.reduce_sum(tf.multiply(self.deconv_a1_chosen,self.deconv_a1),axis=3)
        self.deconv_a2_pred = tf.reduce_sum(tf.multiply(self.deconv_a2_chosen,self.deconv_a2),axis=3)
        self.deconv_a3_pred = tf.reduce_sum(tf.multiply(self.deconv_a3_chosen,self.deconv_a3),axis=3)
        self.deconv_a4_pred = tf.reduce_sum(tf.multiply(self.deconv_a4_chosen,self.deconv_a4),axis=3)
        
        self.pc_prediction = tf.add_n([tf.squeeze(self.deconv_E),self.deconv_a1_pred,self.deconv_a2_pred,self.deconv_a3_pred,self.deconv_a4_pred])
        self.pc_target = tf.placeholder(shape=[None,20,20],dtype=tf.float32)
        
        
        self.pc_loss = tf.losses.mean_squared_error(self.pc_target,self.pc_prediction)
    
    def _create_base_net(self):
        #average expectation accross all actions
        self.expectation1 =  fully_connected(self.merged_dropout,512,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.expectation2 =  fully_connected(self.expectation1,self.outputdim,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        
        self.a1_advantages1 =  fully_connected(self.merged_dropout,512,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.a1_advantages2 =  fully_connected(self.a1_advantages1,self.outputdim_a1,
            activation_fn=None,weights_initializer=he_normal())
        
        self.a2_advantages1 =  fully_connected(self.merged_dropout,256,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.a2_advantages2 =  fully_connected(self.a2_advantages1,self.outputdim_a2,
            activation_fn=None,weights_initializer=he_normal())
        
        self.a3_advantages1 =  fully_connected(self.merged_dropout,256,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.a3_advantages2 =  fully_connected(self.a3_advantages1,self.outputdim_a3,
            activation_fn=None,weights_initializer=he_normal())
        
        self.a4_advantages1 =  fully_connected(self.merged_dropout,512,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.a4_advantages2 =  fully_connected(self.a4_advantages1,self.outputdim_a4,
            activation_fn=None,weights_initializer=he_normal())


        
        self.expectation2 = tf.reshape(self.expectation2,[-1,self.num_offsets,self.num_measurements[1]])
        self.a1_advantages2 = tf.reshape(self.a1_advantages2,[-1,self.a_size[0],self.num_offsets,self.num_measurements[1]])
        self.a2_advantages2 = tf.reshape(self.a2_advantages2,[-1,self.a_size[1],self.num_offsets,self.num_measurements[1]])
        self.a3_advantages2 = tf.reshape(self.a3_advantages2,[-1,self.a_size[2],self.num_offsets,self.num_measurements[1]])
        self.a4_advantages2 = tf.reshape(self.a4_advantages2,[-1,self.a_size[3],self.num_offsets,self.num_measurements[1]])


        
        #batch normalize, expectation stream handles the average
        self.a1_advantages2 = self.a1_advantages2 - tf.reduce_mean(self.a1_advantages2,axis=1,keep_dims=True)
        self.a2_advantages2 = self.a2_advantages2 - tf.reduce_mean(self.a2_advantages2,axis=1,keep_dims=True)
        self.a3_advantages2 = self.a3_advantages2 - tf.reduce_mean(self.a3_advantages2,axis=1,keep_dims=True)
        self.a4_advantages2 = self.a4_advantages2 - tf.reduce_mean(self.a4_advantages2,axis=1,keep_dims=True)


        
        #tensor of 0 and 1 which indicate whether the action was taken
        #when we multiply each with the advantage stream and reduce_sum we get only the chosen actions
        self.a1_chosen = tf.placeholder(shape=[None,self.a_size[0],self.num_offsets,self.num_measurements[1]],dtype=tf.float32)
        self.a2_chosen = tf.placeholder(shape=[None,self.a_size[1],self.num_offsets,self.num_measurements[1]],dtype=tf.float32)
        self.a3_chosen = tf.placeholder(shape=[None,self.a_size[2],self.num_offsets,self.num_measurements[1]],dtype=tf.float32)
        self.a4_chosen = tf.placeholder(shape=[None,self.a_size[3],self.num_offsets,self.num_measurements[1]],dtype=tf.float32)

              
        self.a1_pred = tf.reduce_sum(tf.multiply(self.a1_advantages2,self.a1_chosen),axis=1)
        self.a2_pred = tf.reduce_sum(tf.multiply(self.a2_advantages2,self.a2_chosen),axis=1)
        self.a3_pred = tf.reduce_sum(tf.multiply(self.a3_advantages2,self.a3_chosen),axis=1)
        self.a4_pred = tf.reduce_sum(tf.multiply(self.a4_advantages2,self.a4_chosen),axis=1)

        
        #sum up all contributions to the output prediction
        self.prediction = tf.add_n([self.expectation2,self.a1_pred,self.a2_pred,self.a3_pred,self.a4_pred])
            
        #This is the actual
        self.target = tf.placeholder(shape=[None,self.num_offsets,self.num_measurements[1]],dtype=tf.float32)
        
        #Loss function
        self.loss_main  = tf.losses.mean_squared_error(self.target,self.prediction)
    
    def _create_loss_train_ops(self):

        if self.using_pc:
            self.loss_combined = self.pc_weight*self.pc_loss + self.main_weight*self.loss_main
        else:
            self.pc_loss = 0
            self.loss_combined = self.main_weight*self.loss_main
        
        learning_rate = tf.train.exponential_decay(self.start_lr,self.steps,self.half_lr_every_n_steps,0.5)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                     beta1=0.95,
                                     beta2=0.999,
                                     epsilon = 1e-4)
        
        
        #Get & apply gradients from network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients = tf.gradients(self.loss_combined,global_vars)
        self.var_norms = tf.global_norm(global_vars)
        grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,1)
        self.apply_grads = self.trainer.apply_gradients(list(zip(grads,global_vars)))
 