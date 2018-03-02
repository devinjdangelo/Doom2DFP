import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
from skimage import transform



def action_indecies_to_tensor(batch_indecies,num_actions,pc=False):
  #batch_indecies is array of action indecies
  #which indicate which action of the 4 categories was chosen
  #num_actions indicates the total actions in stream to be converted
  #this helper function converts this array into a tensor of 0s and 1s
  #which can be multiplied with the action stream tensor outputs to
  #select the relevent action prediction to be compared with actual
  #action. Output shape must be (?,num_actions,num_offsets,num_measurements)

  #print(batch_indecies)  
  #print(num_actions)
    n_batches = len(batch_indecies)
    if not pc:
        num_offsets = 6
        num_measurements = 7
        out_tensor = np.zeros(shape=(n_batches,num_actions,num_offsets,num_measurements),dtype=np.float32)
        for batch,action_chosen in enumerate(batch_indecies):
            out_tensor[batch,action_chosen,:,:] = 1
  
    else:
        out_tensor = np.zeros(shape=[n_batches,20,20,num_actions])
        for batch,action_chosen in enumerate(batch_indecies):
            out_tensor[batch,:,:,action_chosen] = 1
            
    return out_tensor


def compute_circles_visited(xcords,ycords,verbose=False):
     #this is the most important reward
     #it will reward the agent based on the total area
     #explored. this will be imputed using the xy cordinates
     #mindistance is in map units... 256 is approx 2x the width of a normal hallway
     mindistance = 128 
     #A = (mindistance/2)**2 * 3.14 #circles of radius 1/2 mindistance
     mindistance = mindistance**2 #distance is squared so we don't have to square root in distance formula
        
     coords = np.asarray(list(zip(xcords,ycords)))
     keepcoords = coords
     i = 0
     while i < keepcoords.shape[0]-1:
         refc = keepcoords[i]
         distance = np.sum((keepcoords - refc)**2,axis=1)
         keepcoords = np.asarray([j for n, j in enumerate(keepcoords) if distance[n] > mindistance or n<=i])
         i += 1    
     if verbose:
        print("Over ",coords.shape[0]," Tics, you traveled to ",len(keepcoords)," unique circles.")
     #area = A * len(keepcoords)
     #return area
     return len(keepcoords) #number of unique circles should be sufficient info

        

doom2_monsters = ["Zombieman","ShotgunGuy","ChaingunGuy","WolfensteinSS","DoomImp","Demon","Spectre","LostSoul",
"BetaSkull","Cacodemon","BaronOfHell","HellKnight","Revenant","Arachnotron","Fatso","PainElemental",
"Archvile","Cyberdemon","SpiderMastermind","CommanderKeen","BossBrain","BossEye","BossTarget","SpawnShot","SpawnFire"]

#I measured empiracly the approximate maximum angle of aim between
#monster and player which would still usually score a hit
#So if player fires and angle <= max_angle_to_hit(distance)
#I will say that player successfully "fired at enemy"

distances = [5000,10000,20000,30000,60000,124160,150000,400000]
max_angles = [24,12,8,6,4,3,2,0.5]
max_angles = [2*theta for theta in max_angles]
max_angle_to_hit = lambda distance: np.interp(distance,distances,max_angles)

def detect_hits(labels,agent,melee=False):
  #detects if agent scored a hit with a hitscan weapon
  #assuming he fires successfully in the current state
  #each label in labels has label.object_position_x
  #y and z. Also label.object_name can be compared to
  #doom2_monsters to see if the object is an enemy.
  #agent[0] = xpos. agent[1]=ypos, agent[2]=angle

      monster_angles = []
      distances = []
      for label in labels:
          if label.object_name in doom2_monsters:
              x = label.object_position_x - agent[0]
              y = label.object_position_y - agent[1]
              angle = np.angle(x + 1j*y,deg=True)
              dist = x**2+y**2
              if angle<0:
                  angle += 360
                  
              monster_angles.append(angle)
              distances.append(dist)
      
      if melee:
          maxdist = 81**2 #can't hit regardless of angle if >81 units away
          monster_angles = [monster_angles[i] for i,dist in enumerate(distances) if dist<=maxdist]
          distances = [distances[i] for i,dist in enumerate(distances) if dist<=maxdist]
          
      if len(monster_angles)>0:
            #convert distances to angle tolerances (your aim can be more off when you are close)
         angle_tolerance = max_angle_to_hit(distances)
         hit_differences = np.absolute(monster_angles - agent[2])
         hits = [diff<tol for diff in hit_differences for tol in angle_tolerance]
         any_hit = np.amax(hits)
      else:
         any_hit = False
    
      return any_hit


def get_f(m,offsets):
    f = np.zeros([len(m),len(offsets),m.shape[1]])
    for i,offset in enumerate(offsets):
        f[:-offset,i,:] = m[offset:,:] - m[:-offset,:]
        if i > 0:
            f[-offset:,i,:] = f[-offset:,i-1,:]
    return f

def get_pc_target(sbuff,local_means=False):
    #calculate pixel control targets from most recent 2 frames
    #sbuff 100x160x3x2 CEILAB images (stored as 100x160x6)
    #output is 25x25
    
    cropped_diff = np.absolute(sbuff[2:-2,2:-2,3:] - sbuff[2:-2,2:-2,:3])
    #numpy and skimage implementation which are slightly different... not sure why
    if local_means:
        targets = transform.downscale_local_mean(cropped_diff,[4,4,3])[:,:,0]
    else:
        cropped_diff = np.reshape(cropped_diff,[4,20,4,20,3])
        targets = np.mean(cropped_diff,axis=(0,2,4))
        
    return targets
