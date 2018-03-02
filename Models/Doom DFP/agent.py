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
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,10])
        
               
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
        self.g = [0.4,0,1,0.25,0.25,0.5,0.5]

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
        self.timeout_steps = 150
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
            
        self.objective_weights1 = np.tile(self.measurement_weights,len(self.move_actions)*(len(offsets)-2)).reshape(len(self.move_actions),len(offsets)-2,self.num_measurements[1])
        self.objective_weights2 = np.tile(self.measurement_weights,len(self.jump_actions)*(len(offsets)-2)).reshape(len(self.jump_actions),len(offsets)-2,self.num_measurements[1])
        self.objective_weights3 = np.tile(self.measurement_weights,len(self.use_actions)*(len(offsets)-2)).reshape(len(self.use_actions),len(offsets)-2,self.num_measurements[1])
        self.objective_weights4 = np.tile(self.measurement_weights,len(self.attack_actions)*(len(offsets)-2)).reshape(len(self.attack_actions),len(offsets)-2,self.num_measurements[1])
        
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
            map_num = randint(1,self.number_of_maps)
            #map_num = 1 #override to test 
            if map_num<10:
                str_map_num = "0" + str(map_num)
            elif map_num<100:
                str_map_num = str(map_num)
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
                
            self.timeout_steps = int(min(100 * 2**(self.episode_count/8000),400))

    def update_experience_memory(self,rollout):
        #takes a random sample of 1/4th of the experiences from the episode
        #and stores them into memory.
        
       
        rollout = np.reshape(rollout,[-1,10])
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
            a2 = action_indecies_to_tensor(a_idx_array[:,1],self.action_group_lengths[1])
            a3 = action_indecies_to_tensor(a_idx_array[:,2],self.action_group_lengths[2])
            a4 = action_indecies_to_tensor(a_idx_array[:,3],self.action_group_lengths[3])
            
            m_in_prepped = np.stack(exp_batch[:,2],axis=0)
            m_in_prepped = self.prep_m(np.copy(m_in_prepped[:,:num_observe_measurements]),levels=True,verbose=False)
            
            target_m_prepped = np.stack(exp_batch[:,4],axis=0)
            target_m_prepped = target_m_prepped[:,0,:,:]
            target_m_prepped = self.prep_m(np.copy(target_m_prepped),levels=False,verbose=False)
            
            ahists = np.stack(exp_batch[:,5],axis=0)

            memstate1 = np.stack(exp_batch[:,6],axis=0)
            memstate2 = np.stack(exp_batch[:,7],axis=0)
            memstate3 = np.stack(exp_batch[:,8],axis=0)
            memstate4 = np.stack(exp_batch[:,9],axis=0)
            
            observations = np.stack(exp_batch[:,0],axis=0)
                                
            
            feed_dict = {self.local_DFP.observation:observations,
                self.local_DFP.measurements:m_in_prepped,
                self.local_DFP.action_history:ahists,
                self.local_DFP.a1_chosen:a1,
                self.local_DFP.a2_chosen:a2,
                self.local_DFP.a3_chosen:a3,
                self.local_DFP.a4_chosen:a4,
                self.local_DFP.target:target_m_prepped,
                self.local_DFP.goals:np.vstack(exp_batch[:,3]),
                self.local_DFP.memcache_l1:memstate1,
                self.local_DFP.memcache_l2:memstate2,
                self.local_DFP.memcache_l3:memstate3,
                self.local_DFP.memcache_l4:memstate4,
                self.local_DFP.episodes:self.episode_count}
            loss,g_n,v_n,_ = sess.run([self.local_DFP.loss,
                self.local_DFP.grad_norms,
                self.local_DFP.var_norms,
                self.local_DFP.apply_grads],feed_dict=feed_dict)
            return loss, g_n,v_n
        else:
            return 0,0,0
    
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
            out_tensors = [self.local_DFP.merged_input2,self.local_DFP.mem1_dense1,
                        self.local_DFP.mem2_dense1,self.local_DFP.mem3_dense1,
                        self.local_DFP.a1_advantages2,self.local_DFP.a2_advantages2,
                        self.local_DFP.a3_advantages2,self.local_DFP.a4_advantages2]

            present,mem1,mem2,mem3,a1_dist, a2_dist, a3_dist, a4_dist = sess.run(out_tensors, 
            feed_dict={
            self.local_DFP.observation:[s4],
            self.local_DFP.measurements:[m_prepped],
            self.local_DFP.goals:[self.measurement_weights],
            self.local_DFP.action_history:[ahistory],
            self.local_DFP.exploring:explore,
            self.local_DFP.memcache_l1:[self.memcache_1],
            self.local_DFP.memcache_l2:[self.memcache_2],
            self.local_DFP.memcache_l3:[self.memcache_3],
            self.local_DFP.memcache_l4:[self.memcache_4],
            self.local_DFP.episodes:self.episode_count})

            present = np.reshape(present,[1,512])
            self.memcache_1 = np.append(present,self.memcache_1[:4,:],axis=0)
            if self.episode_steps % 5 == 0:
                mem1 = np.reshape(mem1,[1,256])
                self.memcache_2 = np.append(mem1,self.memcache_2[:5,:],axis=0)
                if self.episode_steps % 25 == 0:
                    mem2 = np.reshape(mem2,[1,256])
                    self.memcache_3 = np.append(mem2,self.memcache_3[:5,:],axis=0)
                    if self.episode_steps % 125 == 0:
                        mem3 = np.reshape(mem3,[1,256])
                        self.memcache_4 = np.append(mem3,self.memcache_4[:5,:],axis=0)
                        
                             
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
                self.just_pressed_attack = True
                if self.selected_weapon[-1]==2:
                    self.attack_cooldown = 3
                elif self.selected_weapon[-1]==3:
                    self.attack_cooldown = 7
                elif self.selected_weapon[-1]==1:
                    self.attack_cooldown = 3
            elif a4==0:
                self.attack_cooldown = 0 #doing nothing has no cooldown
                

        #action_array is an action accepted by Vizdoom engine
        a = np.asarray([a1,a2,a3,a4])
        action_array = np.concatenate((self.move_actions[a1],self.jump_actions[a2],self.use_actions[a3],self.attack_actions[a4])).tolist()
        return a,action_array

    def process_m(self,m_raw):
        
        
        #n_weapons = sum(m_raw[8:17]) / 4.5 - 1
        
        self.selected_weapon = [self.selected_weapon[-1],m_raw[1]]
        self.selected_ammo = [self.selected_ammo[-1],m_raw[0]]
        self.successfully_fired_shot = (self.selected_weapon[1]==self.selected_weapon[0]) and (self.selected_ammo[1] < self.selected_ammo[0])
        
        fist_active = 1 if self.selected_weapon[-1]==1 else 0
        pistol_active = 1 if self.selected_weapon[-1]==2 else 0
        shotgun_active = 1 if self.selected_weapon[-1]==3 else 0

        #weap1 = 1 if fist only and =2 if fist and chainsaw
        weapon1 = 1 if m_raw[9]>1 else 0
        #weap2 = 1 if pistol
        weapon2 = 1 if m_raw[10]>0 else 0
        #weap3 = 1 if shotty and =2 if also super shotty
        weapon3 = 1 if m_raw[11]>0 else 0

        #ammo2 = pistol bullets
        ammo2 = m_raw[20]
        #ammo3 = shotgun shells
        ammo3 = m_raw[21]
        
        health = m_raw[2]
        armor = m_raw[3]
        
        #all_kills includes monsters killing other monsters which can be very confusing
        #in the early stages of training/exploring as the agent will get 3-6 kills totally randomly
        self.all_kills.append(m_raw[5] )
        
        items = m_raw[4]  
        
       
        self.episode_xpos.append(m_raw[6])
        self.episode_ypos.append(m_raw[7])
        if len(self.episode_xpos) > 1:

            area_explored = compute_circles_visited(self.episode_xpos,self.episode_ypos,verbose=False)
            self.episode_explored.append(area_explored)

            #labels has info about visible objects including enemies (used for hit detection)
            labels = self.state.labels
            agent = [self.episode_xpos[-1],self.episode_ypos[-1],m_raw[29]]
            using_melee = True if fist_active else False
            hit_scored = detect_hits(labels,agent,melee=using_melee)

            if hit_scored and self.attack_action_in_progress==3:
                #if aiming close to visible enemy and attack action in progress we score a "hit"
                self.hits += 1
                self.last_hit_n_ago = 0
                
            if self.last_hit_n_ago<=3:
                #if within 3 steps we scored a "hit" and an enemy dies we score a "kill"
                self.last_hit_n_ago+=1
                current_kills = self.all_kills[-1] - self.all_kills[-2]
                self.direct_kills = self.direct_kills + current_kills 

        else: 
            area_explored = 0
            dist_traveled = 0
            
        m = [weapon1,weapon2,weapon3,fist_active,pistol_active,
            shotgun_active,health,armor,self.direct_kills,ammo2,ammo3,
            self.hits,area_explored]
        #m = [health,kills,pistol_ammo]

        return m        
    
    def prep_m(self,m,levels=False,verbose=False):
        #takes numpy array (?,num_measurements) and normalizes for network
        #can normalize in levels (i.e. for input to M) or changes (i.e. for output target)
        
    
        if levels:
            #measurements represent running totals or current value in case of health
            m = np.reshape(m,[-1,num_observe_measurements])
            m[:,0] = m[:,0]/2           #weap1
            m[:,1] = m[:,1]/1           #weap2
            m[:,2] = m[:,2]/1           #weap3
            m[:,3] = m[:,3]/1           #fist
            m[:,4] = m[:,4]/1           #pistol
            m[:,5] = m[:,5]/1           #shotgun
            m[:,6] = m[:,6]/50 - 1      #health
            m[:,7] = m[:,7]/50 - 1      #armor
            m[:,8] = m[:,8]/10 - 1      #kills
            m[:,9] = m[:,9]/40 - 1      #ammo2
            m[:,10] = m[:,10]/10 - 1      #ammo3

            
            if verbose:
                print("range level Weapon1: ", np.amin(m[:,0])," to ",np.amax(m[:,0]))
                print("range level Weapon2: ",np.amin(m[:,1]), " to ",np.amax(m[:,1]))
                print("range level Weapon3: ",np.amin(m[:,2]), " to ",np.amax(m[:,2]))
                print("range level Health: ", np.amin(m[:,3])," to ",np.amax(m[:,3]))
                print("range level Armor: ",np.amin(m[:,4,]), " to ",np.amax(m[:,4]))
                print("range level Kills: ",np.amin(m[:,5]), " to ",np.amax(m[:,5]))
                print("range level Ammo2: ",np.amin(m[:,6]), " to ",np.amax(m[:,6]))
                print("range level Ammo3: ",np.amin(m[:,7]), " to ",np.amax(m[:,7]))


        else:
            m[:,:,0] = m[:,:,0]/75          #health
            m[:,:,1] = m[:,:,1]/75          #armor
            m[:,:,2] = m[:,:,2]/2 - 1       #kills
            m[:,:,3] = m[:,:,3]/15          #ammo2
            m[:,:,4] = m[:,:,4]/10          #ammo3
            m[:,:,5] = m[:,:,5]/20 - 1      #hits needs minus 1
            m[:,:,6] = m[:,:,6]/5 - 1        #explored


            
            
            if verbose:
                print("range delta health: ", np.amin(m[:,:,0])," to ",np.amax(m[:,:,0]))
                print("range delta armor: ",np.amin(m[:,:,1]), " to ",np.amax(m[:,:,1]))
                print("range delta kills: ",np.amin(m[:,:,2]), " to ",np.amax(m[:,:,2]))
                print("range delta ammo2: ", np.amin(m[:,:,3])," to ",np.amax(m[:,:,3]))
                print("range delta ammo3: ",np.amin(m[:,:,4]), " to ",np.amax(m[:,:,4]))
                print("range delta hits: ",np.amin(m[:,:,5]), " to ",np.amax(m[:,:,5]))
                print("range delta explore: ",np.amin(m[:,:,6]), " to ",np.amax(m[:,:,6]))
            
        return m
    
    def work(self,sess,saver,train):
        self.mini_batch_iterations = 0
        self.episode_count = 35099
        total_steps = 0
        prevsteps=0
        start_time = time.time()
        self.total_explored = []
        reset_stats=False
        while True:
            self.selected_ammo = [0,0]
            self.selected_weapon = [0,0]
            self.hits = 0
            self.episode_buffer = []
            episode_frames = []
            episode_finished = False
            self.episode_steps = 1
            self.episode_xpos = []
            self.episode_ypos = []
            self.episode_explored = []
            self.direct_kills = 0
            self.all_kills = [0]
            self.episode_count +=1
            self.memcache_1 = np.zeros(shape=[5,512])
            self.memcache_2 = np.zeros(shape=[6,256])
            self.memcache_3 = np.zeros(shape=[6,256])
            self.memcache_4 = np.zeros(shape=[6,256])
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
            s = skimage.transform.resize(s,(self.xdim,self.ydim,3))
            s = skimage.color.rgb2lab(s)
            s[:,:,0] = s[:,:,0]/50 - 1
            s[:,:,1] = s[:,:,1]/128
            s[:,:,2] = s[:,:,2]/128
            sbuffer = np.stack(([s]*2), axis=2) 
            sbuffer = np.reshape(sbuffer,[120,160,6])

            abuffer = np.zeros([27,12])

            steps_per_sec = (total_steps-prevsteps)//(time.time()-start_time)
            prevsteps = total_steps
            start_time = time.time()

            while episode_finished == False:

                #update experience memory to work with new network
                #we need to remember the state of the memory cache
                #at each experience for traiing
                #very important to pass a copy of m4 and not m4, otherwise mbuffer will be permanently modified
                #s4 = sbuffer[:,:,[0,8,17,26]]
                #m4 = mbuffer[:,[0,7]]
                #a4 = abuffer[:,[0,8,17,26]]
                a,action_chosen = self.choose_action(sbuffer,np.copy(m[:num_observe_measurements]),abuffer)  
                if not recording:
                    self.env.make_action(action_chosen,self.frame_skip)        
                else:
                    self.env.set_action(action_chosen)
                    for _ in range(self.frame_skip):
                        self.env.advance_action()
                        
                self.episode_buffer.append([sbuffer,a,m,self.measurement_weights,np.zeros(len(self.offsets)),abuffer,
                 np.copy(self.memcache_1),np.copy(self.memcache_2),np.copy(self.memcache_3),np.copy(self.memcache_4)])
                
                if self.env.is_episode_finished():
                    episode_finished=True  
                else:
                
                    self.state = self.env.get_state()
                    m_raw = self.state.game_variables
                    m = self.process_m(m_raw)       

                    srgb = self.state.screen_buffer
                    
                    srgb = skimage.transform.resize(srgb,(self.xdim,self.ydim,3))
                    s = skimage.color.rgb2lab(srgb)
                    s[:,:,0] = s[:,:,0]/50 - 1
                    s[:,:,1] = s[:,:,1]/128
                    s[:,:,2] = s[:,:,2]/128
                                        
                    s = np.reshape(s, (self.xdim, self.ydim, 3))

                    sbuffer = np.append(s, sbuffer[:,:, :3], axis=2)

                    abuffer = np.append(np.reshape(action_chosen,[1,12]),abuffer[:26,:],axis=0)
                    
                    if self.test_episode and self.test_counter==self.test_for_n:
                        srgb = srgb[:,:,::-1]
                        episode_frames.append(srgb)
                        
                    total_steps += 1 
                    self.episode_steps += 1
                    
                    if self.episode_steps > 3000:
                        #end episode after ~6 minutes
                        episode_finished = True
                    elif self.episode_steps>self.timeout_steps:
                        if self.episode_explored[-1] - self.episode_explored[-self.timeout_steps] == 0 and self.all_kills[-1] - self.all_kills[-self.timeout_steps] == 0:
                            #end episode if we have not explored anywhere new or got any kills for a period of time
                            episode_finished = True
                                       
            self.episode_kills.append(self.direct_kills)
            self.episode_lengths.append(self.episode_steps*4/35)
            self.total_explored.append(self.episode_explored[-1])
            self.total_hits.append(self.hits)


            
            # Update the network using the experience buffer at the end of the episode.
            self.update_experience_memory(self.episode_buffer)
            if train == True and total_steps>20000:
                losses = []
                norms = []
                iterations = (self.mini_batches_per_64exp * self.episode_steps) // 64
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
                mean_explored = np.mean(self.total_explored)
                mean_hits = np.mean(self.total_hits)
                time_per_step = 1/35*4
                self.images = np.array(episode_frames)
                imageio.mimwrite(self.gif_path+'/image'+str(self.episode_count)+'.gif',self.images,duration=time_per_step)
                savelist = [self.episode_count,total_steps,mean_length,mean_kills,mean_hits,mean_explored,steps_per_sec]
                with open('teststats8700.csv', 'a') as myfile:
                    wr = csv.writer(myfile)
                    wr.writerow(['{:.2f}'.format(x) for x in savelist])
                
                reset_stats=True
                
            if  self.episode_count % self.train_stats_every_n==0 and total_steps>30000:
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
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                self.summary_writer.add_summary(summary, self.episode_count)     
                self.summary_writer.flush()
                
                reset_stats=True
                
            print("episodes: ",self.episode_count,", Total Experiences: ",total_steps,"Steps per second: ",
                  steps_per_sec, "Episode Kills: ", self.episode_kills[-1], "Explored: ", self.total_explored[-1],
                  "Episode Length: ", int(self.episode_lengths[-1]), " seconds",)
            print("Total Hits: ", self.hits, "Episode Goal: ",self.measurement_weights, "Exploration: ", self.exploration,
                  "Testing? " , self.test_episode,"Timeout Steps",self.timeout_steps,"learning rate",3e-4 * 0.5**(self.episode_count/9000)) 
            
            if reset_stats:
                self.episode_kills=[]
                self.episode_lengths = []
                self.total_explored = []
                self.total_hits=[]
                reset_stats=False
            
                
                
                

if __name__ == '__main__':
    
    numactions = 576
    num_total_measurements = 13
    num_observe_measurements = 11 #Number of observed measurements
    num_predict_measurements = 7 #number of predicted measurements
    offsets = [2,4,8,16,32,64] # Set of temporal offsets
    load_model = True #ther to load a saved model
    train = True #Whether to train the network
    model_path = 'C:/Users/djdev/Documents/tensorflow models/doom2_entryway_memory' #Path to save the model to
    gif_path = './frames_goals' #Path to save gifs of agent performance to
    exploration = 'bayesian'
    
    recording = True #enables smooth playback for agent recording

    #frame dimensions
    xdim = 120
    ydim = 160
    
    tf.reset_default_graph()
    
    exp_buff = ExperienceBuffer()
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
        
    with open('teststats8700.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Total Episodes","Total Steps","Length","Kills","Circles Explored","Steps Per Second"])
    
    

    # Create worker classes
    agent = Worker(DoomGame(),model_path,offsets,
            exp_buff,[num_observe_measurements,num_predict_measurements],
            gif_path,exploration,xdim,ydim)
    saver = tf.train.Saver(max_to_keep=5,keep_checkpoint_every_n_hours=2)
    
    with tf.Session() as sess:
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            ckpt = 'C:/Users/djdev/Documents/tensorflow models/doom2_entryway_memory/model-35100.ckpt'
            saver.restore(sess,ckpt)
        else:
            sess.run(tf.global_variables_initializer())
        agent.work(sess,saver,True)
