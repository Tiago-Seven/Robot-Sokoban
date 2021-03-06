import numpy as np
from collections import defaultdict
import itertools
from sokoban import Game
import matplotlib.pyplot as plt
import hashlib

def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
    """ 
    Creates an epsilon-greedy policy based 
    on a given Q-function and epsilon. 
       
    Returns a function that takes the state 
    as an input and returns the probabilities 
    for each action in the form of a numpy array  
    of length of the action space(set of possible actions). 
    """
    def policyFunction(state): 
   
        Action_probabilities = np.ones(num_actions, 
                dtype = float) * epsilon / num_actions 

        best_action = np.argmax(Q[hashlib.sha224(state.data.tobytes()).hexdigest()]) 
        Action_probabilities[best_action] += (1.0 - epsilon) 
        return Action_probabilities 
   
    return policyFunction 

def qLearning(env, num_episodes, discount_factor = 0.5, 
                            alpha = 0.8):

    epsilon = 0.2 
    """ 
    Q-Learning algorithm: Off-policy TD control. 
    Finds the optimal greedy policy while improving 
    following an epsilon-greedy policy"""
       
    # Action value function 
    # A nested dictionary that maps 
    # state -> (action -> action-value). 
    Q = defaultdict(lambda: np.zeros(Game.ACTION_SPACE_SIZE)) 
   
       
    # Create an epsilon greedy policy function 
    # appropriately for environment action space 
    policy = createEpsilonGreedyPolicy(Q, epsilon, Game.ACTION_SPACE_SIZE) 

    rewards = []
    # For every episode 
    for i in range(num_episodes):
        episode_reward = 0.0
        # Reset the environment and pick the first action 
        state = env.reset() 
        first = True
        for e in itertools.count(): 
            # get probabilities of all actions from current state 
            action_probabilities = policy(state) 
            # choose action according to  
            # the probability distribution 
            action = np.random.choice(np.arange( 
                      len(action_probabilities)), 
                       p = action_probabilities) 
   
            # take action and get reward, transit to next state 
            next_state, reward, done = env.step(action) 
            episode_reward += reward
               
            # TD Update 
            best_next_action = np.argmax(Q[hashlib.sha224(next_state.data.tobytes()).hexdigest()])     
            td_target = reward + discount_factor * Q[hashlib.sha224(next_state.data.tobytes()).hexdigest()][best_next_action] 
            td_delta = td_target - Q[hashlib.sha224(state.data.tobytes()).hexdigest()][action] 
            
            Q[hashlib.sha224(state.data.tobytes()).hexdigest()][action] += alpha * td_delta 

            # done is True if episode terminated    
            if done: 
                rewards.append(episode_reward/e)
                break
            state = next_state
        
       
    return Q, rewards
