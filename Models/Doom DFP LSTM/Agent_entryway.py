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
    def __init__(self, max_episodes = 60,max_sequence_length=256):
        #episodes stores all experiences of an epsisode
        #self.episode_lengths tells us how long each episode was
        #max epsiodes is the max episodes we will store at once
        self.episodes = []
        self.episode_lengths = []
        self.episode_scores = []
        self.max_episodes = max_episodes
        
        #how many vars do we store at each step 
        self.experience_dimension = 9
        self.max_sequence_length = max_sequence_length
        
        self.per_smoothing = 0.5 #coefficient for per 
        
    
    def add(self,episode,score):
        if len(self.episodes) >= self.max_episodes:
            self.episodes.pop(0)
            self.episode_lengths.pop(0)
            self.episode_scores.pop(0)
        self.episodes.append(episode)
        self.episode_lengths.append(len(episode))
        self.episode_scores.append(score)
            
    
    def sample(self,num_episodes):
        #create mini batch of sequences of num_episode size
        #each sequence is chosen randomly weighted by length of episode and episode score
        #with replacement
        
        #sequence length is determined dynamically, either length of shortest episode in batch
        #or max_sequence_length. In either case, the sequence length is mapped to nearest multiple
        # of 32 for performance reasons. For episodes longer than sequence length, we will choose a random
        #offset from 0 to length-sequence_length into the start of the episode which decides where the
        #sequence starts within the episode.
        
        #output is numpy array with dim (num_episodes,sequence_length,self.experience_dimension)

        sampled_episodes = np.random.choice(self.episodes,size=num_episodes,replace=True,p=self.episode_probabilities)
        sampled_episodes = [np.array(episode).reshape(-1,self.experience_dimension) for episode in sampled_episodes]

        lengths = [episode.shape[0] for episode in sampled_episodes]
        min_length = min(lengths)
        sequence_length = self.seq_length_to_32(min_length)
        
        offsets = [np.random.randint(0,max(length-sequence_length,1)) for length in lengths]
        sampled_sequences = [sampled_episodes[i][offset:offset+sequence_length,:] for i,offset in enumerate(offsets)]
        #sampled_sequences = [episode[:sequence_length,:] for episode in sampled_episodes]
        sampled_sequences = np.stack(sampled_sequences,axis=0)
        sampled_sequences = np.reshape(sampled_sequences,[-1,self.experience_dimension])
    
        return sampled_sequences,sequence_length
    
    def seq_length_to_32(self,minlength):
        #map sequence length to nearest valid multiple of 32
        #for performance reasons, see: http://svail.github.io/rnn_perf/
        if minlength<32:
            return minlength
        elif minlength>self.max_sequence_length:
            return self.max_sequence_length
        else:
            return 32*(minlength//32)
            
        
    @property
    def episode_probabilities(self):
        #probability to sample each episode
        score_lengths = np.stack([self.episode_scores,self.episode_lengths])
        score_product = np.prod(score_lengths,axis=0)
        score_product = np.power(score_product,self.per_smoothing)
        probabilities = score_product/np.sum(score_product)
        return probabilities.tolist()
        
               
class Worker():
    def __init__(self,game,model_path,offsets,exp_buff,num_measurements,gif_path,exploration,
                 xdim,ydim,using_pc,start_lr,half_lr_every_n_steps,main_weight,pc_weight):
        self.offsets = offsets
        self.exp_buff = exp_buff
        self.model_path = model_path
        self.gif_path = gif_path
        self.episode_kills = []
        self.episode_lengths = []
        self.loss = []
        self.loss_pc = [] 
        self.g_norm = []
        self.attack_cooldown = False
        self.last_hit_n_ago = 10
        self.num_measurements = num_measurements
        self.summary_writer = tf.summary.FileWriter("train_1")
        self.g = [0.25,0,0.1,0.1,1,0.7,0.7]

        self.holding_down_use=0
        self.frame_skip = 4
        self.temp = 0.25
        self.env = game
        self.start_game(init=True)

        self.xdim = xdim
        self.ydim = ydim

        self.local_DFP = DFP_Network(self.action_group_lengths,len(offsets),num_measurements,self.xdim,self.ydim,
                                     using_pc,start_lr,half_lr_every_n_steps,main_weight,pc_weight)

        self.exploration = exploration
        self.just_pressed_attack = False
        self.number_of_maps = 3
        self.timeout_steps = 400
        self.train_stats_every_n = 25
        self.test_episode = False
        self.test_every_n = 100
        self.test_for_n = 10
        self.test_counter = 1

        self.total_hits=[]

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
            
        self.objective_weights1 = np.tile(self.measurement_weights,len(self.move_actions)*(len(offsets)-2)).reshape(len(self.move_actions),len(offsets)-2,self.num_measurements[1])
        self.objective_weights2 = np.tile(self.measurement_weights,len(self.jump_actions)*(len(offsets)-2)).reshape(len(self.jump_actions),len(offsets)-2,self.num_measurements[1])
        self.objective_weights3 = np.tile(self.measurement_weights,len(self.use_actions)*(len(offsets)-2)).reshape(len(self.use_actions),len(offsets)-2,self.num_measurements[1])
        self.objective_weights4 = np.tile(self.measurement_weights,len(self.attack_actions)*(len(offsets)-2)).reshape(len(self.attack_actions),len(offsets)-2,self.num_measurements[1])
        

    
    def start_game(self,init=False):
        if init:
            self.env.load_config("doom2.cfg")
            #forward back strafe l,r turn l,r and speed 54 valid combinations
            moven = 7
            self.move_actions = [list(a) for a in it.product([0, 1], repeat=moven)]
            self.move_actions = [a for a in self.move_actions if a[0]+a[1]<2 and a[2]+a[3]<2 
                                and a[5]+a[6]<2]
            #jump 2 valid combinations
            jumpn = 1
            self.jump_actions = [list(a) for a in it.product([0, 1], repeat=jumpn)]
            #use 2 valid combinatoins
            usen = 1
            self.use_actions = [list(a) for a in it.product([0, 1], repeat=usen)]

            #switch next or prev weapon or attack (only one at a time 4 valid combinations)
            attackn = 3
            self.attack_actions = [list(a) for a in it.product([0, 1], repeat=attackn)]
            self.attack_actions = [a for a in self.attack_actions if a[0]+a[1]+a[2]<2]
            
            #turn left or right (only one can't do both 3 possibilities) 
            #turnn = 2
            #self.turn_actions = [list(a) for a in it.product([0,1], repeat=turnn)]
            #self.turn_actions = [a for a in self.turn_actions if a[0] + a[1] < 2]


            n_move_actions = len(self.move_actions)
            n_jump_actions = len(self.jump_actions)
            n_use_actions = len(self.use_actions)
            n_attack_actions = len(self.attack_actions)
            #n_turn_actions = len(self.turn_actions)
            self.action_group_lengths = [n_move_actions,n_jump_actions,n_use_actions,n_attack_actions]
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
            self.map_num = 1
            
            if self.map_num<10:
                str_map_num = "0" + str(self.map_num)
            elif self.map_num<100:
                str_map_num = str(self.map_num)
            mapstr = "map" + str_map_num
             
            self.env.set_doom_map(mapstr)
            self.env.new_episode()
 
            altx_spawns = [-537.573,719.645,179.128,390.992,795.696,192.956]
            alty_spawns = [697.594,1591.212,1568.438,1129.852,553.442,1775.975]
 
            numaltspawns = 6
            altspawnnum = randint(1,numaltspawns+1)
            if altspawnnum < numaltspawns:
                xcoord = altx_spawns[altspawnnum-1]
                ycoord = alty_spawns[altspawnnum-1]
                warp_command = "warp " + str(xcoord) + " " + str(ycoord)
                self.env.send_game_command(warp_command)
             
            if np.random.uniform()>0.5:
                self.env.send_game_command("give shotgun")
            
            self.timeout_steps = int(400 - 300*0.5**(self.total_steps/400000))

    def update_experience_memory(self,rollout):
        #takes a random sample of 1/4th of the experiences from the episode
        #and stores them into memory.
        
       
        rollout = np.reshape(rollout,[-1,9])
        measurements = np.stack(rollout[:,2],axis=0)
        
        #score episodes by (1+kills)*(explored) such that 1% increase in either -> ~1% increase in score
        #but neither can be 0. We will prioritize replay episodes with higher scores
        score = (measurements[-1,4]) + (measurements[-1,6])/3
        
        m_present = measurements[:,(-num_predict_measurements):]
        targets = get_f(m_present,self.offsets) #Generate targets using measurements and offsets
        rollout[:,4] = list(zip(targets))
                
        rollout[:-1,6] = rollout[1:,6]
        #probably don't want to train on the last observation but might be ok
        rollout[-1,6] = np.zeros([20,20])
        #size = len(rollout) // 4
        #rollout = np.array(random.sample(rollout,size))
        
        rollout[1:,7:] = rollout[:-1,7:]
        rollout[0,7] = np.zeros((1, self.local_DFP.lstm1.state_size.c), np.float32)
        rollout[0,8] = np.zeros((1, self.local_DFP.lstm1.state_size.h), np.float32)
        
        self.exp_buff.add(list(zip(rollout)),score)
        

    def train(self,sess,batch_size):
        #Get a batch of experiences from the buffer and use them to update the global network
        #to filter down to the chosen actions by batch.
        
        exp_batch,sequence_length = self.exp_buff.sample(batch_size)
        a_idx_array = np.vstack(exp_batch[:,1])
        

        #print(a_idx_array)
        a1 = action_indecies_to_tensor(a_idx_array[:,0],self.action_group_lengths[0])
        a2 = action_indecies_to_tensor(a_idx_array[:,1],self.action_group_lengths[1])
        a3 = action_indecies_to_tensor(a_idx_array[:,2],self.action_group_lengths[2])
        a4 = action_indecies_to_tensor(a_idx_array[:,3],self.action_group_lengths[3])
        
        m_in_prepped = np.stack(exp_batch[:,2],axis=0)
        m_in_prepped = self.prep_m(np.copy(m_in_prepped[:,:num_observe_measurements]),levels=True,verbose=False)
        
        target_m_prepped = np.stack(exp_batch[:,4],axis=0)
        target_m_prepped = target_m_prepped[:,0,:,:]
        target_m_prepped = self.prep_m(np.copy(target_m_prepped),levels=False,verbose=False)
        
        ahists = np.stack(exp_batch[:,5],axis=0)

        
        observations = np.stack(exp_batch[:,0],axis=0)[:,:,:, :3]
        
        target_pc = np.stack(exp_batch[:,6],axis=0)
        
        pc_a1 = action_indecies_to_tensor(a_idx_array[:,0],self.action_group_lengths[0],pc=True)
        pc_a2 = action_indecies_to_tensor(a_idx_array[:,1],self.action_group_lengths[1],pc=True)
        pc_a3 = action_indecies_to_tensor(a_idx_array[:,2],self.action_group_lengths[2],pc=True)
        pc_a4 = action_indecies_to_tensor(a_idx_array[:,3],self.action_group_lengths[3],pc=True) 

        c_states = np.zeros((batch_size, self.local_DFP.lstm1.state_size.c), np.float32)
        h_states = np.zeros((batch_size, self.local_DFP.lstm1.state_size.h), np.float32)  
        
        #get initial states in the sequence
        #c_states = np.stack(exp_batch[:,7],axis=0)
        #c_states = np.reshape(c_states,[batch_size,sequence_length,self.local_DFP.lstm1.state_size.c])[:,0,:]
        
        #h_states = np.stack(exp_batch[:,8],axis=0)
        #h_states = np.reshape(h_states,[batch_size,sequence_length,self.local_DFP.lstm1.state_size.h])[:,0,:]
                            
        
        feed_dict = {self.local_DFP.observation:observations,
            self.local_DFP.measurements:m_in_prepped,
            self.local_DFP.action_history:ahists,
            self.local_DFP.c_in:c_states,
            self.local_DFP.h_in:h_states,
            self.local_DFP.seq_length:sequence_length,
            self.local_DFP.a1_chosen:a1,
            self.local_DFP.a2_chosen:a2,
            self.local_DFP.a3_chosen:a3,
            self.local_DFP.a4_chosen:a4,
            self.local_DFP.target:target_m_prepped,
            self.local_DFP.goals:np.vstack(exp_batch[:,3]),
            self.local_DFP.steps:self.total_steps}
        
        fetch_list = [self.local_DFP.loss_main,
            self.local_DFP.grad_norms,
            self.local_DFP.var_norms,
            self.local_DFP.apply_grads]
        
        if self.local_DFP.using_pc:
            
            feed_dict.update({self.local_DFP.deconv_a1_chosen:pc_a1,
            self.local_DFP.deconv_a2_chosen:pc_a2,
            self.local_DFP.deconv_a3_chosen:pc_a3,
            self.local_DFP.deconv_a4_chosen:pc_a4,
            self.local_DFP.pc_target:target_pc})
            
            fetch_list.append(self.local_DFP.pc_loss)
            main_loss,pc_loss, g_n,v_n,_ = sess.run(fetch_list,feed_dict=feed_dict)
            return main_loss,pc_loss, g_n,v_n,sequence_length

        else:
            main_loss,g_n,v_n,_ = sess.run(fetch_list,feed_dict=feed_dict)
            return main_loss,g_n,v_n,sequence_length
        
    
    def network_pass_to_actions(self,a1_dist,a2_dist,a3_dist,a4_dist):
        #convert forward pass of network into indecies which indicate
        #which action from each group is most advantageous according to
        #current measurment goal

        a1_pred = a1_dist[0,:,2:,:] * self.objective_weights1 #* self.temporal_discounting1
        a1_pred=np.sum(a1_pred,axis=2)
        a1_pred=np.sum(a1_pred,axis=1)
        a1 = np.argmax(a1_pred)
        
        a2_pred = a2_dist[0,:,2:,:] * self.objective_weights2 #* self.temporal_discounting2
        a2_pred=np.sum(a2_pred,axis=2)
        a2_pred=np.sum(a2_pred,axis=1)
        a2 = np.argmax(a2_pred)
        
        a3_pred = a3_dist[0,:,2:,:] * self.objective_weights3 #* self.temporal_discounting3
        a3_pred=np.sum(a3_pred,axis=2)
        a3_pred=np.sum(a3_pred,axis=1)
        a3 = np.argmax(a3_pred)
        
        a4_pred = a4_dist[0,:,2:,:] * self.objective_weights4 #* self.temporal_disocunting4
        a4_pred=np.sum(a4_pred,axis=2)
        a4_pred=np.sum(a4_pred,axis=1)
        a4 = np.argmax(a4_pred)   

        return a1,a2,a3,a4    

    def choose_action(self,s4,m4,ahistory):
        if self.exploration == 'bayesian':
                        
            explore = not self.test_episode
            
            m_prepped = self.prep_m(m4,levels=True)[0,:]
            out_tensors = [self.local_DFP.lstm_state,self.local_DFP.a1_advantages2,self.local_DFP.a2_advantages2,
                        self.local_DFP.a3_advantages2,self.local_DFP.a4_advantages2]

            lstm_state,a1_dist, a2_dist, a3_dist, a4_dist = sess.run(out_tensors, 
            feed_dict={
            self.local_DFP.observation:[s4[:,:, :3]],
            self.local_DFP.measurements:[m_prepped],
            self.local_DFP.goals:[self.measurement_weights],
            self.local_DFP.action_history:[ahistory],
            self.local_DFP.c_in:self.c_state,
            self.local_DFP.h_in:self.h_state,
            self.local_DFP.exploring:explore,
            self.local_DFP.steps:self.total_steps})      

            self.c_state, self.h_state = lstm_state
                                       
            a1,a2,a3,a4 = self.network_pass_to_actions(a1_dist,a2_dist,a3_dist,a4_dist)
         
        else:
            raise ValueError('Exploration policy,',exploration,
            ', is undefined. Please implement policy in Worker.choose_action')
        

        if self.attack_cooldown>0:
            a4 = self.attack_action_in_progress
            self.attack_cooldown -= 1
            if (a4==1 or a4==2) and self.attack_cooldown==1:
                a4=0 #need to release switch weapon button to be able to switch weapons again on next step!
                self.attack_action_in_progress = 0
            self.just_pressed_attack = False

        else:
            self.attack_action_in_progress = a4

            if a4==1 or a4==2:
                self.attack_cooldown = 8  #on the 9th step after pressing switch weapons, the agent will actually fire if fire is pressed
            elif a4==3:
                #Need to check the selected weapon numbers are correct
                self.just_pressed_attack = True
                if self.selected_weapon==1:
                    self.attack_cooldown = 3
                elif self.selected_weapon==2:
                    self.attack_cooldown = 3
                elif self.selected_weapon==3:
                    self.attack_cooldown = 7
                # elif self.selected_weapon==4:
                #     self.attack_cooldown = 13
                # elif self.selected_weapon==5:
                #     self.attack_cooldown = 1
                # elif self.selected_weapon==6:
                #     self.attack_cooldown = 4
                # elif self.selected_weapon==7:
                #     self.attack_cooldown = 9
                    
                
            elif a4==0:
                self.attack_cooldown = 0
                
            
            # if self.holding_down_use==1:
            #     if self.use_cooldown>0:
            #         a3==1
            #         self.use_cooldown -= 1
            #     else:
            #         self.holding_down_use=0
            #         a3=0
            # elif a3==1:
            #     self.holding_down_use=1
            #     self.use_cooldown = 4
            
        #action_array is an action accepted by Vizdoom engine
        a = np.asarray([a1,a2,a3,a4])
        action_array = np.concatenate((self.move_actions[a1],self.jump_actions[a2],self.use_actions[a3],self.attack_actions[a4])).tolist()
        return a,action_array

    def process_m(self,m_raw):
        
        self.selected_weapon = m_raw[1]
        #ammo2 = pistol bullets
        ammo2 = m_raw[20]
        #ammo3 = shotgun shells
        ammo3 = m_raw[21]
        #ammo4 = rockets
        ammo4 = m_raw[22]
        #ammo5 = cells
        ammo5 = m_raw[23]

        health = m_raw[2]
        armor = m_raw[3]
        
        self.all_kills.append(m_raw[5])
       
        self.episode_xpos.append(m_raw[6])
        self.episode_ypos.append(m_raw[7])
        if len(self.episode_xpos) > 1:

            area_explored = compute_circles_visited(self.episode_xpos,self.episode_ypos,verbose=False)
            self.episode_explored.append(area_explored)

            #labels has info about visible objects including enemies (used for hit detection)
            labels = self.state.labels
            agent = [self.episode_xpos[-1],self.episode_ypos[-1],m_raw[29]]
            using_melee = True if self.selected_weapon==1 else False
            hit_scored = detect_hits(labels,agent,melee=using_melee)

            if hit_scored and self.attack_action_in_progress==3:
                #if aiming close to visible enemy and attack action in progress we score a "hit"
                self.hits += 1
                self.last_hit_n_ago = 0
                
            if self.last_hit_n_ago<=3:
                #if within 3 steps we scored a "hit" and an enemy dies we score a "kill"
                #need to check if 3 is enough for non hitscan weapons
                self.last_hit_n_ago+=1
                current_kills = self.all_kills[-1] - self.all_kills[-2]
                self.direct_kills = self.direct_kills + current_kills 
            

        else: 
            area_explored = 0
            dist_traveled = 0
        
        self.kill_history.append(self.direct_kills)
        #m = [health,armor,ammo2,ammo3,ammo4,ammo5,self.direct_kills,self.hits,area_explored]
        m = [health,armor,ammo2,ammo3,self.direct_kills,self.hits,area_explored]
        
        return m     
    
    def prep_m(self,m,levels=False,verbose=False):
        #takes numpy array (?,num_measurements) and normalizes for network
        #can normalize in levels (i.e. for input to M) or changes (i.e. for output target)
        
    
        if levels:
            #measurements represent running totals or current value in case of health
            m = np.reshape(m,[-1,num_observe_measurements])
            m[:,0] = m[:,0]/50 - 1      #health
            m[:,1] = m[:,1]/50 - 1      #armor
            m[:,2] = m[:,2]/40 - 1      #ammo2
            m[:,3] = m[:,3]/10 - 1      #ammo3
            #m[:,4] = m[:,4]/30 - 1      #ammo4
            #m[:,5] = m[:,5]/100 - 1      #ammo5



            
            if verbose:
                print("range level health: ", np.amin(m[:,0])," to ",np.amax(m[:,0]))
                print("range level armor: ",np.amin(m[:,1]), " to ",np.amax(m[:,1]))
                print("range level ammo2: ", np.amin(m[:,2])," to ",np.amax(m[:,2]))
                print("range level ammo3: ",np.amin(m[:,3]), " to ",np.amax(m[:,3]))



        else:
            #measurements that can go up or down are divided by a constant to roughly map
            #to range -1 to 1. Measurements that can only go up are divided by a constant
            #and 1 is subtracted out to map to range -1 to 1.
            m[:,:,0] = m[:,:,0]/75          #health
            m[:,:,1] = m[:,:,1]/75          #armor
            m[:,:,2] = m[:,:,2]/15          #ammo2
            m[:,:,3] = m[:,:,3]/5          #ammo3
            #m[:,:,4] = m[:,:,4]/4          #ammo4
            #m[:,:,5] = m[:,:,5]/40          #ammo5
            m[:,:,4] = m[:,:,4]/2 - 1       #kills
            m[:,:,5] = m[:,:,5]/20 - 1      #hits 
            m[:,:,6] = m[:,:,6]/5 - 1       #explored


            
            
            if verbose:
                print("range delta health: ", np.amin(m[:,:,0])," to ",np.amax(m[:,:,0]))
                print("range delta armor: ",np.amin(m[:,:,1]), " to ",np.amax(m[:,:,1]))
                print("range delta kills: ",np.amin(m[:,:,2]), " to ",np.amax(m[:,:,2]))
                print("range delta ammo2: ", np.amin(m[:,:,3])," to ",np.amax(m[:,:,3]))
                print("range delta ammo3: ",np.amin(m[:,:,4]), " to ",np.amax(m[:,:,4]))
                print("range delta hits: ",np.amin(m[:,:,5]), " to ",np.amax(m[:,:,5]))
                print("range delta explored: ",np.amin(m[:,:,6]), " to ",np.amax(m[:,:,6]))
            
        return m
        
    def play_episode(self,sess):    
        self.hits = 0
        self.episode_buffer = []
        self.episode_frames = []
        episode_finished = False
        self.episode_steps = 1
        self.episode_xpos = []
        self.episode_ypos = []
        self.episode_explored = []
        self.direct_kills = 0
        self.all_kills = [0]
        self.kill_history = [0]
        self.episode_count +=1
        self.c_state = np.zeros((1, self.local_DFP.lstm1.state_size.c), np.float32)
        self.h_state = np.zeros((1, self.local_DFP.lstm1.state_size.h), np.float32)                  
        self.set_objective_weights()
        self.start_game()
        
        self.state = self.env.get_state()
        m_raw = self.state.game_variables
        m = self.process_m(m_raw)
                    
        s = self.state.screen_buffer
        s = s[:-80,:,:] #crop out the HUD
        s = skimage.transform.resize(s,(self.xdim,self.ydim,3))
        s = skimage.color.rgb2lab(s)
        s[:,:,0] = s[:,:,0]/50 - 1
        s[:,:,1] = s[:,:,1]/128
        s[:,:,2] = s[:,:,2]/128
        sbuffer = np.stack(([s]*2), axis=2) 
        sbuffer = np.reshape(sbuffer,[self.xdim,self.ydim,6])

        abuffer = np.zeros(12)

        while episode_finished == False:

            #update experience memory to work with new network
            #we need to remember the state of the memory cache
            #at each experience for traiing
            #very important to pass a copy of m4 and not m4, otherwise mbuffer will be permanently modified
            #s4 = sbuffer[:,:,[0,8,17,26]]
            #m4 = mbuffer[:,[0,7]]
            #a4 = abuffer[:,[0,8,17,26]]
            a,action_chosen = self.choose_action(sbuffer,np.copy(m[:num_observe_measurements]),abuffer)  
            self.env.make_action(action_chosen,self.frame_skip)            
            self.episode_buffer.append([sbuffer,a,m,self.measurement_weights,np.zeros(len(self.offsets)),
                                        abuffer,get_pc_target(sbuffer),np.copy(self.c_state),np.copy(self.h_state)])
            
            if self.env.is_episode_finished():
                episode_finished=True  
            else:
            
                self.state = self.env.get_state()
                m_raw = self.state.game_variables
                m = self.process_m(m_raw)      
                
                srgb = self.state.screen_buffer
                srgb = srgb[:-80,:,:] #crop out the HUD
                srgb = skimage.transform.resize(srgb,(self.xdim,self.ydim,3))
                
                s = skimage.color.rgb2lab(srgb)
                s[:,:,0] = s[:,:,0]/50 - 1
                s[:,:,1] = s[:,:,1]/128
                s[:,:,2] = s[:,:,2]/128
                                    
                #s = np.reshape(s, (self.xdim, self.ydim, 3))

                sbuffer = np.append(s, sbuffer[:,:, :3], axis=2)

                abuffer = action_chosen
                
                if (self.test_episode and (self.episode_count-10)%100==0) or (self.episode_count+10)%100==0:
                    srgb = srgb[:,:,::-1]
                    self.episode_frames.append(srgb)
                    
                self.total_steps += 1 
                self.episode_steps += 1
                
                if self.episode_steps > 3000:
                    #end episode after ~6 minutes
                    episode_finished = True
                elif self.episode_steps>self.timeout_steps:
                    if self.episode_explored[-1] - self.episode_explored[-self.timeout_steps] == 0 and self.kill_history[-1] - self.kill_history[-self.timeout_steps] == 0:
                        #end episode if we have not explored anywhere new or got any kills for a period of time
                        episode_finished = True
                                   

        
    
    def work(self,sess,saver,train):
        self.use_tensorboard = True
        self.mini_batch_iterations = 0
        self.episode_count = 0
        self.total_steps = 0
        start_time = time.time()
        self.total_explored = []
        prevsteps = 0
        start_time = time.time()
        while True:
            self.episode_kills=[]
            self.episode_lengths = []
            self.total_explored = []
            self.total_hits=[]
            self.loss=[]
            self.loss_pc = []
            self.test_episode = True if self.episode_count%100==0 and self.episode_count>0 else False
            for i in range(0,10):
                self.play_episode(sess)
                self.update_experience_memory(self.episode_buffer)
                self.episode_kills.append(self.direct_kills)
                self.episode_lengths.append(self.episode_steps*4/35)
                self.total_explored.append(self.episode_explored[-1])
                self.total_hits.append(self.hits)
                
                steps_per_sec = (self.total_steps-prevsteps)//(time.time()-start_time)
                prevsteps = self.total_steps
                start_time = time.time()
                
                print("episodes: ",self.episode_count,", Total Experiences: ",self.total_steps,"Steps per second: ",
                  steps_per_sec, "Episode Kills: ", self.episode_kills[-1], "Explored: ", self.total_explored[-1],
                  "Episode Length: ", int(self.episode_lengths[-1]), " seconds",)
                print("Total Hits: ", self.hits, "Episode Goal: ",self.measurement_weights, "Exploration: ", self.exploration,
                  "Testing? " , self.test_episode) 
            
            print("Mean Score: ", np.mean(self.exp_buff.episode_scores[-10:]),
                  "Var Score: ", np.std(self.exp_buff.episode_scores[-10:]),
                  "Max Score: ", np.amax(self.exp_buff.episode_scores[-10:]))
            
            if train == True and self.episode_count>=20:
                losses = []
                losses_pc = []
                norms = []
                iters = np.sum(self.exp_buff.episode_lengths[-10:])//64 + 1
                for i in range(0,iters):
                    if self.local_DFP.using_pc:
                        loss_main,loss_pc,g_n,v_n,sequence_length = self.train(sess,4)
                        losses_pc.append(loss_pc)
                        norms.append(g_n)
                        losses.append(loss_main)
                        print("Iter: ", i, "Loss Main: ", loss_main, "Loss_pc: ",loss_pc, "Seq Length: ", sequence_length)
                    else: 
                        loss_main,g_n,v_n,sequence_length = self.train(sess,4)
                        losses.append(loss_main)
                        norms.append(g_n)
                        print("Iter: ", i, "Loss Main: ", loss_main, "Seq Length: ", sequence_length)
                loss = np.mean(losses)
                if self.local_DFP.using_pc:
                    loss_pc = np.mean(losses_pc)
                g_norm = np.mean(norms)
                self.mini_batch_iterations += 1

            
                
            if self.test_episode:
            #save model, test statistics, and gif of test episode after every test episodes
                saver.save(sess,self.model_path+'/model-'+str(self.episode_count)+'.ckpt')
                mean_kills = np.mean(self.episode_kills)
                mean_length = np.mean(self.episode_lengths)
                mean_explored = np.mean(self.total_explored)
                mean_hits = np.mean(self.total_hits)
                time_per_step = 1/35*4
                self.images = np.array(self.episode_frames)
                imageio.mimwrite(self.gif_path+'/image'+str(self.episode_count)+'.gif',self.images,duration=time_per_step)
                savelist = [self.episode_count,self.total_steps,mean_length,mean_kills,mean_hits,mean_explored,self.map_num,steps_per_sec]
                with open('teststats8700.csv', 'a') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow(['{:.2f}'.format(x) for x in savelist])
                
                
            if (self.episode_count+10)%100==0:
            #save gif of exploration episode once every 100 episodes
                time_per_step = 1/35*4
                self.images = np.array(self.episode_frames)
                imageio.mimwrite(self.gif_path+'/image'+str(self.episode_count)+'.gif',self.images,duration=time_per_step)
            
                
            if  self.use_tensorboard and self.episode_count%10==0 and self.episode_count>=20:
                mean_kills = np.mean(self.episode_kills)
                mean_length = np.mean(self.episode_lengths)
                mean_explored = np.mean(self.total_explored)
                mean_hits = np.mean(self.total_hits)
                summary = tf.Summary()
                summary.value.add(tag='Performance/Kills', simple_value=float(mean_kills))
                summary.value.add(tag='Performance/Length', simple_value=float(mean_length))
                summary.value.add(tag='Performance/Exploration', simple_value=float(mean_explored))
                summary.value.add(tag='Performance/hits', simple_value=float(mean_hits))
                if train == True:
                    summary.value.add(tag='Losses/Loss', simple_value=float(loss))
                    if self.local_DFP.using_pc:
                        summary.value.add(tag='Losses/Loss_PC', simple_value=float(loss_pc))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_norm))
                self.summary_writer.add_summary(summary, self.episode_count)     
                self.summary_writer.flush()
                
                


            
                
                
                

if __name__ == '__main__':
    
    numactions = 576
    num_total_measurements = 7
    num_observe_measurements = 4 #Number of observed measurements
    num_predict_measurements = 7 #number of predicted measurements
    offsets = [2,4,8,16,32,64] # Set of temporal offsets
    load_model = False #ther to load a saved model
    train = True #Whether to train the network
    model_path = 'C:/Users/djdev/Documents/tensorflow models/LSTM lessPER' #Path to save the model to
    gif_path = './frames_goals' #Path to save gifs of agent performance to
    exploration = 'bayesian'
    
    use_pc = False #use pixel control auxiliary task?
    starting_learning_rate = 1e-4
    half_learning_rate_every_n_steps = 5000000
    pc_weight = 0
    main_weight = 1

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
        wr.writerow(["Total Episodes","Total Steps","Length","Kills","Hits","Circles Explored","map_num","Steps Per Second"])
    

    # Create worker classes
    agent = Worker(DoomGame(),model_path,offsets,
            exp_buff,[num_observe_measurements,num_predict_measurements],
            gif_path,exploration,xdim,ydim,use_pc,starting_learning_rate,
            half_learning_rate_every_n_steps,main_weight,pc_weight)
    saver = tf.train.Saver(max_to_keep=5,keep_checkpoint_every_n_hours=4)
    
    with tf.Session() as sess:
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        agent.work(sess,saver,True)
