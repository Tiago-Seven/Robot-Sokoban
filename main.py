import sys
import math
import string
import queue
import numpy as np

mode = "play"

if mode == "play": ########################### PLAY ###########################
  import pygame
  from sokoban import Game, start_game, checkSameBox, print_game
  from real_time_display import Basic_Map, Real_Time_Display
  from utils import Move 
  
  pygame.init()
  level = start_game()
  game = Game('levels', level)
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
  load_model_name = "models/1stTest__1576587247.model"



  DISCOUNT = 0.99
  REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
  # Minimum number of steps in a memory to start training
  MIN_REPLAY_MEMORY_SIZE = 1000
  MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
  UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
  MODEL_NAME = '1stTest'
  MIN_REWARD = 0.1  # For model save
  MEMORY_FRACTION = 0.20

  # Environment settings
  EPISODES = 5000

  # Exploration settings
  epsilon = 1  # not a constant, going to be decayed
  EPSILON_DECAY = 0.99975
  MIN_EPSILON = 0.001

  #  Stats settings
  AGGREGATE_STATS_EVERY = 50  # episodes

  # For stats
  ep_rewards = [0]
  
  # For more repetitive results
  np.random.seed(1)
  np.random.seed(1)

  game = Game('levels', 1)
  agent = DQNAgent()

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
    game = Game('levels', 1)
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

      # only model actions
      # action = np.argmax(agent.get_qs(current_state))

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
  EPISODES = 2

  load_model_name = "models/1stTest__1576591913.model"

  agent = DQNAgent()
  agent.model = load_model(load_model_name)

  game = Game('levels', 1)
  pygame.init()
  size = game.load_size()
  screen = pygame.display.set_mode(size)

  moves = []

  DISPLAY_REAL_TIME = True


  for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    game = Game('levels', 1)

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