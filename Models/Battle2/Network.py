import tensorflow as tf
from tensorflow.contrib.slim import conv2d, fully_connected, flatten, dropout
from tensorflow.python.keras.initializers import he_normal
from tensorflow.python.keras.layers import LeakyReLU


class DFP_Network():
    def __init__(self,a_size,num_offsets,num_measurements,xdim,ydim):
        #Inputs and visual encoding layers
        #num_measurements[0] = num_observe_measurements
        #num_measurements[1] = num_predict_measuremnets
        
        self.observation = tf.placeholder(shape=[None,xdim,ydim,4],dtype=tf.float32)
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
        
        self.measurements = tf.placeholder(shape=[None,num_measurements[0]],dtype=tf.float32)
        self.dense_m1 = fully_connected(flatten(self.measurements),128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        self.dense_m2 = fully_connected(self.dense_m1,128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        self.dense_m3 = fully_connected(self.dense_m2,128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        
        self.goals = tf.placeholder(shape=[None,num_measurements[1]],dtype=tf.float32)
        self.dense_g1 = fully_connected(flatten(self.goals),128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        self.dense_g2 = fully_connected(self.dense_g1,128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())
        self.dense_g3 = fully_connected(self.dense_g2,128,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())  
        
        self.merged_input1 = tf.concat([self.dense_m3,self.convout,self.dense_g3],1,name="InputMerge")


            
        #We calculate separate expectation and advantage streams, then combine then later
        #This technique is described in https://arxiv.org/pdf/1511.06581.pdf
        
        outputdim = num_measurements[1]*num_offsets
        outputdim_a1 = num_measurements[1]*num_offsets*a_size[0]
        outputdim_a4 = num_measurements[1]*num_offsets*a_size[1]
        
        #average expectation accross all actions
        self.expectation1 =  fully_connected(self.merged_input1,512,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.expectation2 =  fully_connected(self.expectation1,outputdim,
            activation_fn=None,weights_initializer=he_normal())    

        #split actions into functionally seperable groups
        #e.g. the expectations of movements depend intimately on
        #combinations of movements (e.g. forward left vs forward right)
        #but the expectations of movements can be seperated from the outcome
        #of switching weapons for example. This separation reduces the
        # number of outputs of the model by an order of magnitude or more
        #when the number of subactions is large while maintaining the ability
        #to choose from a large number of actions.
        
        self.a1_advantages1 =  fully_connected(self.merged_input1,512,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.a1_advantages2 =  fully_connected(self.a1_advantages1,outputdim_a1,
            activation_fn=None,weights_initializer=he_normal())

        self.a4_advantages1 =  fully_connected(self.merged_input1,512,
            activation_fn=LeakyReLU(0.2),weights_initializer=he_normal())    
        self.a4_advantages2 =  fully_connected(self.a4_advantages1,outputdim_a4,
            activation_fn=None,weights_initializer=he_normal())


        
        self.expectation2 = tf.reshape(self.expectation2,[-1,num_offsets,num_measurements[1]])
        self.a1_advantages2 = tf.reshape(self.a1_advantages2,[-1,a_size[0],num_offsets,num_measurements[1]])
        self.a4_advantages2 = tf.reshape(self.a4_advantages2,[-1,a_size[1],num_offsets,num_measurements[1]])


        
        #batch normalize, expectation stream handles the average
        self.a1_advantages2 = self.a1_advantages2 - tf.reduce_mean(self.a1_advantages2,axis=1,keep_dims=True)
        self.a4_advantages2 = self.a4_advantages2 - tf.reduce_mean(self.a4_advantages2,axis=1,keep_dims=True)


        
        #tensor of 0 and 1 which indicate whether the action was taken
        #when we multiply each with the advantage stream and reduce_sum we get only the chosen actions
        self.a1_chosen = tf.placeholder(shape=[None,a_size[0],num_offsets,num_measurements[1]],dtype=tf.float32)
        self.a4_chosen = tf.placeholder(shape=[None,a_size[1],num_offsets,num_measurements[1]],dtype=tf.float32)

              
        self.a1_pred = tf.reduce_sum(tf.multiply(self.a1_advantages2,self.a1_chosen),axis=1)
        self.a4_pred = tf.reduce_sum(tf.multiply(self.a4_advantages2,self.a4_chosen),axis=1)

        
        #sum up all contributions to the output prediction
        self.prediction = tf.add_n([self.expectation2,self.a1_pred,self.a4_pred])
            
        #This is the actual
        self.target = tf.placeholder(shape=[None,num_offsets,num_measurements[1]],dtype=tf.float32)
        
        #Loss function
        self.loss = tf.reduce_sum(tf.squared_difference(self.prediction,self.target))
        
        self.episodes = tf.placeholder(shape=(),dtype=tf.int32)
        starting_learning_rate = 1e-4
        learning_rate = tf.train.exponential_decay(starting_learning_rate,self.episodes,20000,0.5)
        
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                     beta1=0.95,
                                     beta2=0.999,
                                     epsilon = 1e-4)
        
        
        #Get & apply gradients from network
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.gradients = tf.gradients(self.loss,global_vars)
        self.var_norms = tf.global_norm(global_vars)
        grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,5)
        self.apply_grads = self.trainer.apply_gradients(list(zip(grads,global_vars)))
 