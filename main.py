import sys
import math
import string
import queue
import numpy as np
import os

os.environ["PYTHONHASHSEED"] = "0"
mode = "q_autonomous"

if mode == "play": ########################### PLAY ###########################
  import pygame
  from sokoban import Game, start_game, checkSameBox, print_game
  from real_time_display import Basic_Map, Real_Time_Display
  from utils import Move 
  
  collisions = True

  pygame.init()
  level = start_game()
  game = Game('training_levels', level)
  size = game.load_size()
  screen = pygame.display.set_mode(size)
  
  moves = [Move((0,0),(0,0),"")]

  DISPLAY_REAL_TIME = True

  if DISPLAY_REAL_TIME:
    basic_map = Basic_Map(game.get_matrix())
    real_time_display = Real_Time_Display(basic_map)
    real_time_display.run(moves, collisions)
  else:
    size = game.load_size()
    screen = pygame.display.set_mode(size)
      
  move_array = []
  boxes_moves = []
    
  while 1:
    if game.is_completed():
      if len(move_array) > 0:
        real_time_display.run(move_array, collisions)
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
    if len(moves) > 0 and DISPLAY_REAL_TIME:
        if(len(moves) > 1):
          boxes_moves.append(moves[1])

        if(checkSameBox(boxes_moves)):
          real_time_display.run(move_array, collisions)
          real_time_display.run(moves, collisions)
          move_array = []
          boxes_moves =  []
          moves = []
            
        for move in moves:
          move_array.append(move)

        moves = []
        
        if(game.index == 0):
          real_time_display.run(move_array, collisions)
          move_array = []
          boxes_moves =  []
            
    if not DISPLAY_REAL_TIME:
      print_game(game.matrix, screen)

    pygame.display.update()
elif mode == "train": ########################### TRAIN DEEP ###########################
  #Adapted from https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/  
  import time
  import matplotlib.pyplot as plt
  from tqdm import tqdm

  from sokoban import Game
  from DeepRL import DQNAgent

  ## INPUT
  load = False
  load_model_name = "models/lvl9_new_deep_learning_1000__1578158969.model"

  MODEL_NAME = 'level9'
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
  if (load):
    from keras.models import load_model
    agent.model = load_model(load_model_name)
    agent.target_model.set_weights(agent.model.get_weights())

  # Iterate over episodes
  for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0.0
    step = 1

    # Reset environment and get initial state
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
      else:
          # Get random action
          action = np.random.randint(0, game.ACTION_SPACE_SIZE)

      new_state, reward, done = game.step(action)
      # Transform new continous state to new discrete state and count reward
      episode_reward += reward

      # Every step we update replay memory and train main network
      agent.update_replay_memory(
          (current_state, action, reward, new_state, done))
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
elif mode == "autonomous": ########################### AUTONOMOUS DEEP ###########################
  from keras.models import load_model
  import pygame
  from tqdm import tqdm

  from sokoban import Game, start_game, checkSameBox, print_game
  from real_time_display import Basic_Map, Real_Time_Display
  from utils import Move 

  from DeepRL import DQNAgent
  EPISODES = 10

  load_model_name = "models/lvl9_new_deep_learning_1000__1578158969.model"
  collisions = True
  agent = DQNAgent()
  agent.model = load_model(load_model_name)

  game = Game('training_levels', 9)
  pygame.init()
  size = game.load_size()
  screen = pygame.display.set_mode(size)

  moves = []

  DISPLAY_REAL_TIME = True


  for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    level=9
    game = Game('training_levels', level)

    if DISPLAY_REAL_TIME:
      basic_map = Basic_Map(game.get_matrix())
      real_time_display = Real_Time_Display(basic_map)
      real_time_display.run(moves, collisions)
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
                real_time_display.run(move_array, collisions)
                real_time_display.run(moves, collisions)
                move_array = []
                boxes_moves =  []
                moves = []
              
            for move in moves:
                move_array.append(move)
            moves = []
            if(game.index == 0):
                print(move_array)
                real_time_display.run(move_array, collisions)
                move_array = []
                boxes_moves =  []
              
        if not DISPLAY_REAL_TIME:
            print_game(game.matrix, screen)

        pygame.display.update()
        if game.is_completed():
            print("done!")
            done=True

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done=True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done=True
elif mode == "q_learning": ########################### TRAIN Q-LEARNING ###########################
    from sokoban import Game, start_game, checkSameBox, print_game
    from QLearning import qLearning, createEpsilonGreedyPolicy
    import matplotlib.pyplot as plt
    import json
    import pygame
    from tqdm import tqdm 
    from real_time_display import Basic_Map, Real_Time_Display
    from utils import Move 
    import hashlib

    run_name = "level6"
    level = 6
    game = Game('training_levels', level)
    Q, stats = qLearning(game, 30000)

    # writing
    np.save(run_name, np.array(dict(Q)))
    plt.plot(stats)
    plt.show()

    policy = createEpsilonGreedyPolicy(Q, 0, Game.ACTION_SPACE_SIZE)

    EPISODES = 10

    game = Game('training_levels', level)
    pygame.init()
    size = game.load_size()
    screen = pygame.display.set_mode(size)

    moves = []

    DISPLAY_REAL_TIME = True


    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        game = Game('training_levels', level)

        if DISPLAY_REAL_TIME:
            basic_map = Basic_Map(game.get_matrix())
            real_time_display = Real_Time_Display(basic_map)
            real_time_display.run(moves, collisions)
        else:
            size = game.load_size()
            screen = pygame.display.set_mode(size)

        move_array = []
        boxes_moves = []
        done = False

        while not done:
            print(Q[hashlib.sha224(game.get_state().data.tobytes()).hexdigest()])
            action = np.argmax(Q[hashlib.sha224(game.get_state().data.tobytes()).hexdigest()])
            moves = game.action(action)
            if len(moves) > 0 and DISPLAY_REAL_TIME:
                if(len(moves) > 1):
                    boxes_moves.append(moves[1])

                if(checkSameBox(boxes_moves)):
                    real_time_display.run(move_array, collisions)
                    real_time_display.run(moves, collisions)
                    move_array = []
                    boxes_moves =  []
                    moves = []
                
                for move in moves:
                    move_array.append(move)
                moves = []
                if(game.index == 0):
                    print(move_array)
                    real_time_display.run(move_array, collisions)
                    move_array = []
                    boxes_moves =  []
                
            if not DISPLAY_REAL_TIME:
                print_game(game.matrix, screen)

            pygame.display.update()
            if game.is_completed():
                print("done!")
                done=True

            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done=True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            done=True
elif mode == "q_autonomous": ########################### AUTONOMOUS Q-LEARNING ###########################
    import pygame
    from tqdm import tqdm
    from sokoban import Game, start_game, checkSameBox, print_game
    from real_time_display import Basic_Map, Real_Time_Display
    from utils import Move
    import hashlib
    import time   

    const_hash = hashlib.sha256()

    model_name = "q_tables/level19.npy"
    level = 19

    collisions = True

    Q = np.load(model_name)
    Q = Q.item()
  

    EPISODES = 1

    game = Game('training_levels', level)
    pygame.init()
    size = game.load_size()
    screen = pygame.display.set_mode(size)

    moves = []

    DISPLAY_REAL_TIME = True


    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        game = Game('training_levels', level)

        if DISPLAY_REAL_TIME:
            basic_map = Basic_Map(game.get_matrix())
            real_time_display = Real_Time_Display(basic_map)
            real_time_display.run(moves, collisions)
        else:
            size = game.load_size()
            screen = pygame.display.set_mode(size)

        move_array = []
        boxes_moves = []
        done = False
        steps = 0
        start_time = time.time()
        while not done:
            print(Q[hashlib.sha224(game.get_state().data.tobytes()).hexdigest()])
            action = np.argmax(Q[hashlib.sha224(game.get_state().data.tobytes()).hexdigest()])
            
            if action == 0:
                print("up")
            elif action == 1:
                print("down")
            elif action == 2:
                print("left")
            elif action == 3:
                print("right")
            elif action == 4:
                print("stop")

            steps += 1
            moves = game.action(action)
            if len(moves) > 0 and DISPLAY_REAL_TIME:
                
                if(len(moves) > 1):
                    boxes_moves.append(moves[1])

                if(checkSameBox(boxes_moves)):
                    real_time_display.run(move_array, collisions)
                    real_time_display.run(moves, collisions)
                    move_array = []
                    boxes_moves =  []
                    moves = []
                
                for move in moves:
                    move_array.append(move)
                moves = []
                if(game.index == 0):
                    print(move_array)
                    real_time_display.run(move_array, collisions)
                    move_array = []
                    boxes_moves =  []
                
            if not DISPLAY_REAL_TIME:
                print_game(game.matrix, screen)

            pygame.display.update()
            if game.is_completed():
                if len(move_array) > 0:
                  real_time_display.run(move_array, collisions)
                print("done true")
                done=True


            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done=True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            done=True
        #Elapsed Time and Steps
        elapsed_time = time.time() - start_time
        print("time: {}".format(elapsed_time))
        print("steps: {}".format(steps))