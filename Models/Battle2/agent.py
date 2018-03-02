import imageio
import time
import numpy as np
import tensorflow as tf
import skimage as skimage
from skimage import transform, color
import csv
from random import randint
from math import sqrt

import itertools as it

from vizdoom import *

from Network import DFP_Network
from utils import *


class ExperienceBuffer():
    def __init__(self, buffer_size = 20000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(list(self.buffer)) + len(list(experience)) >= self.buffer_size:
            self.buffer[0:(len(list(experience))+len(list(self.buffer)))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
        
               
class Worker():
    def __init__(self,game,model_path,offsets,exp_buff,num_measurements,gif_path,exploration,xdim,ydim):
        self.offsets = offsets
        self.exp_buff = exp_buff
        self.model_path = model_path
        self.gif_path = gif_path
        self.episode_kills = []
        self.episode_lengths = []
        self.loss = []
        self.g_norm = []
        self.attack_cooldown = False
        self.last_hit_n_ago = 10
        self.num_measurements = num_measurements
        self.summary_writer = tf.summary.FileWriter("train_1")
        self.g = [0.5,0.5,1]

        self.frame_skip = 4
        self.temp = 0.25
        self.env = game
        self.start_game(init=True)

        self.xdim = xdim
        self.ydim = ydim

        self.local_DFP = DFP_Network(self.action_group_lengths,len(offsets),num_measurements,self.xdim,self.ydim)
        
        self.exploration = exploration
        self.just_pressed_attack = False
        self.number_of_maps = 1
        self.timeout_steps = 525
        self.train_stats_every_n = 25
        self.test_episode = False
        self.test_every_n = 100
        self.test_for_n = 10
        self.test_counter = 1

        self.total_hits=[]
        #if set to 32 or higher over-fitting to recent experiences will likely occur
        #not sure how much lower it needs to be
        self.mini_batches_per_64exp = 1
        
        #reshape objective weights 
        self.set_objective_weights(init=True)
        


    def set_objective_weights(self,init=False):
        #set objective weights to test weights or random training weights
        
        if self.test_episode:
            self.measurement_weights = self.g
        else:
            if np.random.uniform()>.75:
                self.measurement_weights = np.random.rand(self.num_measurements[1])
            else:
                self.measurement_weights = self.g
            
        self.objective_weights1 = np.tile(self.measurement_weights,len(self.move_actions)*(len(offsets)-3)).reshape(len(self.move_actions),len(offsets)-3,self.num_measurements[1])
        self.objective_weights4 = np.tile(self.measurement_weights,len(self.attack_actions)*(len(offsets)-3)).reshape(len(self.attack_actions),len(offsets)-3,self.num_measurements[1])
        
        #put more weight on closer predictions than later in objective function
#        if init==True:
#            discounts = []
#            for i in range(0,self.num_measurements[1]-1):
#                weight = 0.99 ** (offsets[i])
#                discounts.append(weight)
#            print(discounts)
#            self.temporal_discounting1 = np.tile(discounts,len(self.move_actions)*(len(offsets)-4)).reshape(len(self.move_actions),len(offsets)-4,self.num_measurements[1])
#            self.temporal_discounting2 = np.tile(discounts,len(self.jump_actions)*(len(offsets)-4)).reshape(len(self.jump_actions),len(offsets)-4,self.num_measurements[1])
#            self.temporal_discounting3 = np.tile(discounts,len(self.use_actions)*(len(offsets)-4)).reshape(len(self.use_actions),len(offsets)-4,self.num_measurements[1])
#            self.temporal_discounting4 = np.tile(discounts,len(self.attack_actions)*(len(offsets)-4)).reshape(len(self.attack_actions),len(offsets)-4,self.num_measurements[1])

    
    def start_game(self,init=False):
        if init:
            self.env.load_config("battle2.cfg")
            #forward back strafe l,r turn l,r and speed 54 valid combinations
            moven = 7
            self.move_actions = [list(a) for a in it.product([0, 1], repeat=moven)]
            self.move_actions = [a for a in self.move_actions if a[0]+a[1]<2 and a[2]+a[3]<2 
                                and a[5]+a[6]<2]

            #attack or not (only one at a time 4 valid combinations)
            attackn = 1
            self.attack_actions = [list(a) for a in it.product([0, 1], repeat=attackn)]
            
            n_move_actions = len(self.move_actions)
            n_attack_actions = len(self.attack_actions)
            #n_turn_actions = len(self.turn_actions)
            self.action_group_lengths = [n_move_actions,n_attack_actions]
            #gives a total of 864 valid combinations (catesian product of all sub action groups -> len=18*2*2*4*3)
            #but only a total of 29 outputs from the neural network (sum of outputs of each group -> 18+2+2+4+3=29)
            
            #self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
            #self.actions = [a for a in self.actions if a[0]+a[1]<2 and a[2]+a[3]<2 and 
             #               a[7]+a[8]<2 and a[9]+a[10]+a[11]<2] 
            
            #e.g. can't turn left and right at the same time
            print("Starting worker ")
            self.env.set_labels_buffer_enabled(True)
            self.env.set_console_enabled(True)
            #self.env.set_ticrate(10)
            self.env.init()
        else:
            map_num = randint(1,self.number_of_maps)
            #map_num = 1 #override to test 
            if map_num<10:
                str_map_num = "0" + str(map_num)
            elif map_num<100:
                str_map_num = str(map_num)
            mapstr = "map" + str_map_num
            
            self.env.set_doom_map(mapstr)
            self.env.new_episode()


    def update_experience_memory(self,rollout):
        #takes a random sample of 1/4th of the experiences from the episode
        #and stores them into memory.
        
       
        rollout = np.reshape(rollout,[-1,5])
        measurements = np.stack(rollout[:,2],axis=0)
        m_present = measurements[:,(-num_predict_measurements):]
        targets = get_f(m_present,self.offsets) #Generate targets using measurements and offsets
        rollout[:,4] = list(zip(targets))
        
        #size = len(rollout) // 4
        #rollout = np.array(random.sample(rollout,size))
        
        self.exp_buff.add(list(zip(rollout)))
        

    def train(self,sess):
        #Get a batch of experiences from the buffer and use them to update the global network
        #to filter down to the chosen actions by batch.
        if len(self.exp_buff.buffer) > 64:
            exp_batch = self.exp_buff.sample(64)
            a_idx_array = np.vstack(exp_batch[:,1])
            #print(a_idx_array)
            a1 = action_indecies_to_tensor(a_idx_array[:,0],self.action_group_lengths[0])
            a4 = action_indecies_to_tensor(a_idx_array[:,1],self.action_group_lengths[1])
            
            m_in_prepped = np.stack(exp_batch[:,2],axis=0)
            m_in_prepped = self.prep_m(np.copy(m_in_prepped[:,:num_observe_measurements]),levels=True,verbose=False)
            
            target_m_prepped = np.stack(exp_batch[:,4],axis=0)
            target_m_prepped = target_m_prepped[:,0,:,:]
            target_m_prepped = self.prep_m(np.copy(target_m_prepped),levels=False,verbose=False)          
            
            observations = np.stack(exp_batch[:,0],axis=0)
                                
            
            feed_dict = {self.local_DFP.observation:observations,
                self.local_DFP.measurements:m_in_prepped,
                self.local_DFP.a1_chosen:a1,
                self.local_DFP.a4_chosen:a4,
                self.local_DFP.target:target_m_prepped,
                self.local_DFP.goals:np.vstack(exp_batch[:,3]),
                self.local_DFP.episodes:self.episode_count}
            loss,g_n,v_n,_ = sess.run([self.local_DFP.loss,
                self.local_DFP.grad_norms,
                self.local_DFP.var_norms,
                self.local_DFP.apply_grads],feed_dict=feed_dict)
            return loss, g_n,v_n
        else:
            return 0,0,0
    
    def network_pass_to_actions(self,a1_dist,a4_dist):
        #convert forward pass of network into indecies which indicate
        #which action from each group is most advantageous according to
        #current measurment goal

        a1_pred = a1_dist[0,:,3:,:] * self.objective_weights1 #* self.temporal_discounting1
        a1_pred=np.sum(a1_pred,axis=2)
        a1_pred=np.sum(a1_pred,axis=1)
        a1 = np.argmax(a1_pred)
             
        a4_pred = a4_dist[0,:,3:,:] * self.objective_weights4 #* self.temporal_disocunting4
        a4_pred=np.sum(a4_pred,axis=2)
        a4_pred=np.sum(a4_pred,axis=1)
        a4 = np.argmax(a4_pred)   

        return a1,a4    

    def choose_action(self,s4,m4):
        if self.exploration == 'e-greedy':
                        
            exp_thresh  = 1 - 0.5 ** (self.episode_count/4000) 
            if not self.test_episode and np.random.uniform()>exp_thresh:
                a1 = np.random.randint(0,high=len(self.move_actions))
                a4 = np.random.randint(0,high=len(self.attack_actions))
            else:
            
                m_prepped = self.prep_m(m4,levels=True)[0,:]
                out_tensors = [self.local_DFP.a1_advantages2,self.local_DFP.a4_advantages2]

                a1_dist,a4_dist = sess.run(out_tensors, 
                feed_dict={
                self.local_DFP.observation:[s4],
                self.local_DFP.measurements:[m_prepped],
                self.local_DFP.goals:[self.measurement_weights],
                self.local_DFP.episodes:self.episode_count})
                   
                a1,a4 = self.network_pass_to_actions(a1_dist,a4_dist)

         
        else:
            raise ValueError('Exploration policy,',exploration,
            ', is undefined. Please implement policy in Worker.choose_action')
        

        if self.attack_cooldown>0:
            a4 = self.attack_action_in_progress
            self.attack_cooldown -= 1

        else:
            self.attack_action_in_progress = a4
            if a4==0:
                self.attack_cooldown = 0
            else:
                self.attack_cooldown = 3
                

        #action_array is an action accepted by Vizdoom engine
        a = np.asarray([a1,a4])
        action_array = np.concatenate((self.move_actions[a1],self.attack_actions[a4])).tolist()
        return a,action_array

    def process_m(self,m_raw):
        

        #ammo2 = pistol bullets
        ammo2 = m_raw[20]
        
        health = m_raw[2]
        
        #all_kills includes monsters killing other monsters which can be very confusing
        #in the early stages of training/exploring as the agent will get 3-6 kills totally randomly
        kills = m_raw[5]
        self.all_kills.append(kills)      
       
            
        m = [health,ammo2,kills]
        #m = [health,kills,pistol_ammo]

        return m        
    
    def prep_m(self,m,levels=False,verbose=False):
        #takes numpy array (?,num_measurements) and normalizes for network
        #can normalize in levels (i.e. for input to M) or changes (i.e. for output target)
        
    
        if levels:
            #measurements represent running totals or current value in case of health
            m = np.reshape(m,[-1,num_observe_measurements])
            m[:,0] = m[:,0]/50 - 1          #health
            m[:,1] = m[:,1]/30 - 1           #ammo


            
            if verbose:
                print("range level health: ", np.amin(m[:,0])," to ",np.amax(m[:,0]))
                print("range level ammo: ",np.amin(m[:,1]), " to ",np.amax(m[:,1]))



        else:
            m[:,:,0] = m[:,:,0]/75          #health
            m[:,:,1] = m[:,:,1]/10           #ammo
            m[:,:,2] = m[:,:,2]/2 - 1       #kills



            
            
            if verbose:
                print("range delta health: ", np.amin(m[:,:,0])," to ",np.amax(m[:,:,0]))
                print("range delta armor: ",np.amin(m[:,:,1]), " to ",np.amax(m[:,:,1]))
                print("range delta ammo: ",np.amin(m[:,:,2]), " to ",np.amax(m[:,:,2]))

        return m
    
    def work(self,sess,saver,train):
        self.mini_batch_iterations = 0
        self.episode_count = 83800
        total_steps = 0
        prevsteps=0
        start_time = time.time()
        self.total_explored = []
        reset_stats=False
        while True:
            self.hits = 0
            self.episode_buffer = []
            episode_frames = []
            episode_finished = False
            self.episode_steps = 1
            self.episode_xpos = []
            self.episode_ypos = []
            self.episode_explored = []
            self.all_kills = [0]
            self.episode_count +=1
            #every 50 episodes test for 10 episodes (test means 0 epsilon greedy exploration and set objective weights)
            if self.test_episode:
                self.test_counter += 1
                if self.test_counter > self.test_for_n:
                    self.test_counter = 1
                    self.test_episode = False
            else:
                self.test_episode = (self.episode_count % self.test_every_n == 0)
                      
            self.set_objective_weights()
            self.start_game()
            
            self.state = self.env.get_state()
            m_raw = self.state.game_variables
            m = self.process_m(m_raw)
                        
            s = self.state.screen_buffer
            s = skimage.transform.resize(s,(self.xdim,self.ydim))
            s = np.reshape(s,(self.xdim, self.ydim)) * 2 - 1

            sbuffer = np.stack(([s]*4), axis=2) 
            sbuffer = np.reshape(sbuffer,[self.xdim,self.ydim,4])

            steps_per_sec = (total_steps-prevsteps)//(time.time()-start_time)
            prevsteps = total_steps
            start_time = time.time()

            while episode_finished == False:

                a,action_chosen = self.choose_action(sbuffer,np.copy(m[:num_observe_measurements]))  
                self.env.make_action(action_chosen,self.frame_skip)        

                self.episode_buffer.append([sbuffer,a,m,self.measurement_weights,np.zeros(len(self.offsets))])
                
                if self.env.is_episode_finished():
                    episode_finished=True  
                else:
                
                    self.state = self.env.get_state()
                    m_raw = self.state.game_variables
                    m = self.process_m(m_raw)       

                    s = self.state.screen_buffer
                    
                    s = skimage.transform.resize(s,(self.xdim,self.ydim))
                    s = np.reshape(s,(self.xdim, self.ydim,1)) * 2 - 1

                    sbuffer = np.append(s, sbuffer[:,:, :3], axis=2)
                    
                    if self.test_episode and self.test_counter==self.test_for_n:
                        episode_frames.append(s)
                        
                    total_steps += 1 
                    self.episode_steps += 1
                    
                    if self.episode_steps > 525:
                        #end episode after ~60 seconds
                        episode_finished = True

                                       
            self.episode_kills.append(self.all_kills[-1])
            self.episode_lengths.append(self.episode_steps*4/35)


            
            # Update the network using the experience buffer at the end of the episode.
            self.update_experience_memory(self.episode_buffer)
            if train == True and total_steps>20000:
                losses = []
                norms = []
                iterations = (self.mini_batches_per_64exp * self.episode_steps) // 64 + 1
                for i in range(1,int(iterations)):
                    loss,g_n,v_n = self.train(sess)
                    losses.append(loss)
                    norms.append(g_n)
                self.loss.append(np.mean(losses))
                self.g_norm.append(np.mean(norms))
                self.mini_batch_iterations += iterations
                print("Avg Loss: ", self.loss[-1], "Average g_norm: ", self.g_norm[-1],
                "Total Iterations: ", self.mini_batch_iterations)
            
                
            # Periodically save gifs of episodes, model parameters, and summary statistics.
            if self.episode_count % 100 == 0 and train == True:
                saver.save(sess,self.model_path+'/model-'+str(self.episode_count)+'.ckpt')
                print("Saved Model")

            if self.test_episode and self.test_counter==self.test_for_n:
                mean_kills = np.mean(self.episode_kills)
                mean_length = np.mean(self.episode_lengths)
                time_per_step = 1/35*4
                self.images = np.array(episode_frames)
                imageio.mimwrite(self.gif_path+'/image'+str(self.episode_count)+'.gif',self.images,duration=time_per_step)
                savelist = [self.episode_count,total_steps,mean_length,mean_kills,steps_per_sec]
                with open('teststats8700.csv', 'a') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow(['{:.2f}'.format(x) for x in savelist])
                
                reset_stats=True
                
            if  self.episode_count % self.train_stats_every_n==0 and total_steps>22000:
                mean_kills = np.mean(self.episode_kills)
                mean_length = np.mean(self.episode_lengths)
                summary = tf.Summary()
                summary.value.add(tag='Performance/Kills', simple_value=float(mean_kills))
                summary.value.add(tag='Performance/Length', simple_value=float(mean_length))
                if train == True:
                    summary.value.add(tag='Losses/Loss', simple_value=float(self.loss[-1]))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(self.g_norm[-1]))
                self.summary_writer.add_summary(summary, self.episode_count)     
                self.summary_writer.flush()
                
                reset_stats=True
                
            print("episodes: ",self.episode_count,", Total Experiences: ",total_steps,"Steps per second: ",
                  steps_per_sec, "Episode Kills: ", self.episode_kills[-1],
                  "Episode Length: ", int(self.episode_lengths[-1]), " seconds",)
            print("Episode Goal: ",self.measurement_weights, "Exploration: ", self.exploration,
                  "Testing? " , self.test_episode,"Timeout Steps",self.timeout_steps,"learning rate",3e-4 * 0.5**(self.episode_count/9000)) 
            
            if reset_stats:
                self.episode_kills=[]
                self.episode_lengths = []
                self.total_explored = []
                self.total_hits=[]
                self.loss=[]
                self.g_norm=[]
                reset_stats=False
            
                
                
                

if __name__ == '__main__':
    
    numactions = 576
    num_total_measurements = 3
    num_observe_measurements = 2 #Number of observed measurements
    num_predict_measurements = 3 #number of predicted measurements
    offsets = [1,2,4,8,16,32] # Set of temporal offsets
    load_model = True #ther to load a saved model
    train = True #Whether to train the network
    model_path = 'C:/Users/djdev/Documents/tensorflow models/battle2' #Path to save the model to
    gif_path = './frames_goals' #Path to save gifs of agent performance to
    exploration = 'e-greedy'
    
    recording = True #enables smooth playback for agent recording

    #frame dimensions
    xdim = 84
    ydim = 84
    
    tf.reset_default_graph()
    
    exp_buff = ExperienceBuffer()
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
        
    with open('teststats8700.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Total Episodes","Total Steps","Length","Kills","Steps Per Second"])
    
    

    # Create worker classes
    agent = Worker(DoomGame(),model_path,offsets,
            exp_buff,[num_observe_measurements,num_predict_measurements],
            gif_path,exploration,xdim,ydim)
    saver = tf.train.Saver(max_to_keep=5,keep_checkpoint_every_n_hours=2)
    
    with tf.Session() as sess:
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        agent.work(sess,saver,True)
