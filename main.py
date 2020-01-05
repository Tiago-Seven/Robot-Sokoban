import sys
import math
import string
import queue
import numpy as np

mode = "q_learning"

if mode == "play": ########################### PLAY ###########################
  import pygame
  from sokoban import Game, start_game, checkSameBox, print_game
  from real_time_display import Basic_Map, Real_Time_Display
  from utils import Move 
  
  pygame.init()
  level = start_game()
  game = Game('training_levels', level)
  size = game.load_size()
  screen = pygame.display.set_mode(size)
  

  moves = []

  DISPLAY_REAL_TIME = False

  if DISPLAY_REAL_TIME:
    basic_map = Basic_Map(game.get_matrix())
    real_time_display = Real_Time_Display(basic_map)
    real_time_display.run(moves)
  else:
    size = game.load_size()
    screen = pygame.display.set_mode(size)
      
  move_array = []
  boxes_moves = []
    
  while 1:
    if game.is_completed():
      break
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit(0)
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
          moves = game.action(0)
        elif event.key == pygame.K_DOWN:
          moves = game.action(1)
        elif event.key == pygame.K_LEFT:
          moves = game.action(2)
        elif event.key == pygame.K_RIGHT:
          moves = game.action(3)
        elif event.key == pygame.K_f:
          moves = game.action(4)
        elif event.key == pygame.K_q:
          sys.exit(0)
        # elif event.key == pygame.K_d:
        #     game.unmove()
    if len(moves) > 0 and DISPLAY_REAL_TIME:
        if(len(moves) > 1):
          boxes_moves.append(moves[1])

        if(checkSameBox(boxes_moves)):
          real_time_display.run(move_array)
          real_time_display.run(moves)
          move_array = []
          boxes_moves =  []
          moves = []
            
        for move in moves:
          move_array.append(move)

        moves = []
        
        if(game.index == 0):
          print(move_array)
          real_time_display.run(move_array)
          move_array = []
          boxes_moves =  []
            
    if not DISPLAY_REAL_TIME:
      print_game(game.matrix, screen)

    pygame.display.update()
elif mode == "train": ########################### TRAIN ###########################
  import time
  import matplotlib.pyplot as plt
  from tqdm import tqdm

  from sokoban import Game
  from DeepRL import DQNAgent

  ## INPUT
  load = False
  load_model_name = "models/multipleMaps__1576623075.model"

  MODEL_NAME = 'level10_changed_discount'
  MIN_REWARD = 0.01  # For model save

  # Environment settings
  EPISODES = 8000

  # Exploration settings
  epsilon = 1  # not a constant, going to be decayed
  EPSILON_DECAY = 0.9996 #99975
  MIN_EPSILON = 0.001

  #  Stats settings
  AGGREGATE_STATS_EVERY = 200  # episodes

  # For stats
  ep_rewards = []
  # chosen_moves = []
  # For more repetitive results
  np.random.seed(2)

  agent = DQNAgent()
    #   from keras.utils import plot_model
    #   import os
    #   os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    #   plot_model(agent.model, to_file='NN_model.png', show_shapes=True,)
    #   exit()
  if (load):
    from keras.models import load_model
    agent.model = load_model(load_model_name)

  # Iterate over episodes
  for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    # agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0.0
    step = 1

    # Reset environment and get initial state
    # level = np.random.randint(1, 3)
    level=9
    game = Game('training_levels', level)
    current_state = game.get_state()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
      # This part stays mostly the same, the change is to query a model for Q values
      if np.random.random() > epsilon:
          # Get action from Q table
          action = np.argmax(agent.get_qs(current_state))
          # chosen_moves.append(action)
          # print("above action")
          # print(action)
          # print("below action")
      else:
          # Get random action
          action = np.random.randint(0, game.ACTION_SPACE_SIZE)

      new_state, reward, done = game.step(action)
      # Transform new continous state to new discrete state and count reward
      episode_reward += reward

      # if SHOW_PREVIEW and episode > 4990:
      #     print_game(game.matrix,screen)
      #     pygame.display.update()
          # print(game.matrix)

      # Every step we update replay memory and train main network
      agent.update_replay_memory(
          (current_state, action, reward, new_state, done))
      # print(current_state)
      # print(new_state)
      agent.train(done, step)

      current_state = new_state
      step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward/step)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(
            ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        # agent.tensorboard.update_stats(
        #     reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(
                'models/{}__{}.model'.format(
                    MODEL_NAME, int(time.time())
                ))

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

  plt.plot(ep_rewards)
  plt.show()
elif mode == "autonomous": ########################### AUTONOMOUS ###########################
  from keras.models import load_model
  import pygame
  from tqdm import tqdm

  from sokoban import Game, start_game, checkSameBox, print_game
  from real_time_display import Basic_Map, Real_Time_Display
  from utils import Move 

  from DeepRL import DQNAgent
  EPISODES = 10

  load_model_name = "models/level10_changed_discount__1578229746.model"

  agent = DQNAgent()
  agent.model = load_model(load_model_name)

  game = Game('training_levels', 10)
  pygame.init()
  size = game.load_size()
  screen = pygame.display.set_mode(size)

  moves = []

  DISPLAY_REAL_TIME = True


  for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # level = np.random.randint(1, 3)
    level=10
    game = Game('training_levels', level)

    if DISPLAY_REAL_TIME:
      basic_map = Basic_Map(game.get_matrix())
      real_time_display = Real_Time_Display(basic_map)
      real_time_display.run(moves)
    else:
      size = game.load_size()
      screen = pygame.display.set_mode(size)

    move_array = []
    boxes_moves = []
    done = False

    while not done:
        action = np.argmax(agent.get_qs(game.get_state()))
        moves = game.action(action)
        if len(moves) > 0 and DISPLAY_REAL_TIME:
            if(len(moves) > 1):
                boxes_moves.append(moves[1])

            if(checkSameBox(boxes_moves)):
                real_time_display.run(move_array)
                real_time_display.run(moves)
                move_array = []
                boxes_moves =  []
                moves = []
              
            for move in moves:
                move_array.append(move)
            moves = []
            if(game.index == 0):
                print(move_array)
                real_time_display.run(move_array)
                move_array = []
                boxes_moves =  []
              
        if not DISPLAY_REAL_TIME:
            print_game(game.matrix, screen)

        pygame.display.update()
        if game.is_completed():
            print("done true")
            done=True

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done=True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done=True
elif mode == "new_training":
    from tqdm import tqdm
    from keras.optimizers import Adam
    from NewDeepRl import Agent
    from sokoban import Game

    optimizer = Adam(lr=0.01)
    level=9
    environment = Game('training_levels', level)
    agent = Agent(environment, optimizer)

    MIN_REWARD = 0

    batch_size = 32
    num_of_episodes = 100
    timesteps_per_episode = 1000
    agent.q_network.summary()
    
    for e in tqdm(range(0, num_of_episodes), ascii=True, unit='episodes'):
        # Reset the environment
        environment = Game('training_levels', level)
        state = environment.get_state()
        
        # Initialize variables
        reward = 0
        terminated = False

        
        for timestep in range(timesteps_per_episode):
            # Run Action
            action = agent.act(state)
            
            # Take action    
            next_state, reward, terminated = environment.step(action) 
            agent.store(state, action, reward, next_state, terminated)
            
            state = next_state
            
            if terminated:
                agent.align_target_model()
                break
                
            if len(agent.experience_replay) > batch_size:
                agent.retrain(batch_size)
            
            if min_reward >= MIN_REWARD:
                agent.target_network.save(
                    'models/{}__{}.model'.format(
                        MODEL_NAME, int(time.time())
                    ))
elif mode == "q_learning":
    from sokoban import Game, start_game, checkSameBox, print_game
    from QLearning import qLearning, createEpsilonGreedyPolicy
    import json
    run_name = "test_dict"
    level = 13
    game = Game('training_levels', level)
    Q = qLearning(game, 1000)
    
    # writing
    np.save(run_name, np.array(dict(Q)))
    print(len(Q))
    policy = createEpsilonGreedyPolicy(Q, 0, Game.ACTION_SPACE_SIZE)
    import pygame
    from tqdm import tqdm

   
    from real_time_display import Basic_Map, Real_Time_Display
    from utils import Move 

    EPISODES = 10

    # load_model_name = "models/big_map_1_goal_1_robot__1578064903.model"

    game = Game('training_levels', level)
    pygame.init()
    size = game.load_size()
    screen = pygame.display.set_mode(size)

    moves = []

    DISPLAY_REAL_TIME = True


    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # level = np.random.randint(1, 3)
        game = Game('training_levels', level)

        if DISPLAY_REAL_TIME:
            basic_map = Basic_Map(game.get_matrix())
            real_time_display = Real_Time_Display(basic_map)
            real_time_display.run(moves)
        else:
            size = game.load_size()
            screen = pygame.display.set_mode(size)

        move_array = []
        boxes_moves = []
        done = False

        while not done:
            print(Q[hash(game.get_state().data.tobytes())])
            action = np.argmax(Q[hash(game.get_state().data.tobytes())])
            moves = game.action(action)
            if len(moves) > 0 and DISPLAY_REAL_TIME:
                if(len(moves) > 1):
                    boxes_moves.append(moves[1])

                if(checkSameBox(boxes_moves)):
                    real_time_display.run(move_array)
                    real_time_display.run(moves)
                    move_array = []
                    boxes_moves =  []
                    moves = []
                
                for move in moves:
                    move_array.append(move)
                moves = []
                if(game.index == 0):
                    print(move_array)
                    real_time_display.run(move_array)
                    move_array = []
                    boxes_moves =  []
                
            if not DISPLAY_REAL_TIME:
                print_game(game.matrix, screen)

            pygame.display.update()
            if game.is_completed():
                print("done true")
                done=True

            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done=True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            done=True