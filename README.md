# Human Level Control of Doom

Previous papers have demonstrated the effectiveness of deep reinforcement learning techniques in the [Vizdoom environment](http://vizdoom.cs.put.edu.pl/). [Dosovitskiy and Koltun (2016)](https://arxiv.org/pdf/1611.01779.pdf)
successfully trained an agent to navigate a complicated maze while engaging enemies. However their environment is still significantly simplified compared to the full Doom singleplayer experience
intended for human players. For example, there is only 1 type of enemy which dies instantly with only 1 hit. The agent has only 1 weapon and no ability to change weapons and can also not jump
or interact with doors, buttons, or switches. I propose and implement a modification of [Dosovitskiy and Koltun's (2016)](https://arxiv.org/pdf/1611.01779.pdf) framework to train an agent to learn to act
within the doom singleplayer mode intended for human players.

## Methodology

Direct Future Predicition (DFP) converts reinforcement learning into a self supervised learning framework. The agent explores the environment and attempts to predict features of the environment at
various points in the future. Then, the agent observes the actual ground truth values of those measurements in the future and uses them to train a deep neural network. This method is found to
outperform A3C substantially. The network used in [Dosovitskiy and Koltun (2016)](https://arxiv.org/pdf/1611.01779.pdf) is sketched below. 

![Base DFP Network](/illustrations/base_dfp.PNG)

The expectation stream outputs the expected value of the measurements at each output average over all possible actions. The action stream outputs the "advantage" of each action by forcing the
average over all actions at each time step to be 0. I modify the action stream further into independent groups. The idea is that the decision of whether to fire
your weapon at a particular time step should be independent of the choice of which direction to move. If you are aimed at an enemy, you should fire reguardless of how you intend to move next. 
This also dramatically reduces the number of outputs required from the network when we have a large action space. In full doom single player, you need at least 12 buttons to perform all required
actions. That gives 4096 combinations of button presses and outputs from a standard DFP network. By splitting the action space into 4 groups, I've reduced this down to 141 outputs from the network
without reducing the total number of actions the agent can take.

Next, I added an LSTM layer between the input streams and the expectation advantage streams. I've also incorporated the "auxiliary unsupervised tasks" of pixel control into the DFP framework. 
[Jaderberg et. al. (2016)](https://arxiv.org/pdf/1611.05397.pdf) found that an A3C agent achieved far superior performance in all tested environments with fewer training examples when augmented 
with auxilliary tasks. In particular, pixel control was found to be the most beneficial. 

One final modification is to provide an additional input which provides the previous action taken by the agent. It is important that the agent remember which buttons it was pressing in recent steps since
this actually changes the behavior of pressing buttons in the current step. For example while using the pistol, if you tap the shoot button you will fire a perfectly acurate bullet in the center of the screen.
If instead you hold down the shoot button, you will fire randomly in a cone. Another example is pressing the "use" button only has any effect when you first press it down. Holding it down will have no effect. 
To have any chance of learning these dependencies, the agent must remember in each step if it was recently pressing a button down.

![Modified DFP Network](/illustrations/modified_dfp.PNG)

## Results

First, I test my modified network on Dosovitskiy and Koltun's (2016) Battle2 scenario. For this test, the only modification I made was to split the action stream into 2 (movement and attack streams). All other
features and hyperparameters were matched as closely as possible to the original implementation. I observe superior performance with <30 million training examples and nearly identical performance with >30 million.
Battle2 scenario has a relatively small action space to doom singleplayer but even still dividing the action space did not adversely affect the final performance of the agent and seems to boost performance
when there are fewer examples to learn from.

![Battle2 Performance](/illustrations/battle.png)

For the first test on full Doom singleplayer, I train an agent only on the first level of Doom 2, Entryway. The first test is DFP augmented only with the action stream split into 4. 
The agent successfully learned to explore the majority of the map and kill enemies with a variety of weapons. 

<img src="/illustrations/entrywaygraph.PNG" width="800">

<img src="/illustrations/entryway.gif" width="400">

I am currently testing DFP modified with pixel control and LSTM. My hope is to obtain a model robust enough to learn many levels simultaneously and perform respectibly on previously unseen levels.

## Code

Requirments:
- Vizdoom
- Tensorflow

The battle2 folder contains all of the code to replicate the test on the battle2 scenario. The DFP folder contains the code to replicate the test on Doom 2 Entryway. Since I am not legally allowed to redistribute
Doom2.wad, you must obtain Doom 2 and place the file doom2.wad in the folder for the code to run. I am currently testing DFP LSTM to find a training regime which works as well or better than DFP on the entryway scenario.
