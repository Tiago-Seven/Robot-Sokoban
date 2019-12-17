#!../bin/python

import sys
import math
import string
import queue
import numpy as np
import random

import pygame
from utils import Move 

background = 255, 226, 191
wall = pygame.image.load('images/wall.png')
floor = pygame.image.load('images/floor.png')
box = pygame.image.load('images/box.png')
box_docked = pygame.image.load('images/box_docked.png')
worker = pygame.image.load('images/worker.png')
worker_docked = pygame.image.load('images/worker_dock.png')
docker = pygame.image.load('images/dock.png')
class Game:
    ACTION_SPACE_SIZE = 5
    
    convert = {
        "#": 0,     # wall
        " ": 0.2,   # floor
        ".": 0.3,   # dock
        "$": 0.5,   # box
        "*": 0.6,   # box on dock
        "+": 0.7,   # worker on dock
        "@": 0.8   # worker on floor
        # current robot 1
    }

    def is_valid_value(self, char):
        if (char == ' ' or  # floor
            char == '#' or  # wall
            char == '@' or  # worker on floor
            char == '.' or  # dock
            char == '*' or  # box on dock
            char == '$' or  # box
                char == '+'):  # worker on dock
            return True
        else:
            return False

    def __init__(self, filename, level, n_robots = 1):
        #if level < 1 or level > 50:
        level = int(level)
        self.queue = queue.LifoQueue()
        self.matrix = []
        self.robots = []
        self.index = 0
        self.episode_step = 0
        self.level = level
        self.n_robots = n_robots
        if level < 1:
            print("ERROR: Level "+str(level)+" is out of range")
            sys.exit(1)
        else:
            file = open(filename, 'r')
            level_found = False
            for line in file:
                row = []
                if not level_found:
                    if "Level "+str(level) == line.strip():
                        level_found = True
                else:
                    if line.strip() != "":
                        row = []
                        for c in line:
                            if c != '\n' and self.is_valid_value(c):
                                row.append(c)
                            elif c == '\n':  # jump to next row when newline
                                continue
                            else:
                                print("ERROR: Level "+str(level) +
                                      " has invalid value "+c)
                                sys.exit(1)
                        self.matrix.append(row)
                    else:
                        break
        self.robots = self.get_robots()
        if len(self.robots) == 0:
            self.robots = self.set_robots()

    def get_robots(self):
        x = 0
        y = 0
        robots = []
        for row in self.matrix:
            for char in row:
                if char == '@':  # robot on floor
                    robots.append((x, y))
                elif char == '+':  # robot on goal
                    robots.append((x, y))
                x = x + 1
            x = 0
            y = y + 1
        return robots

    def set_robots(self):
        robots = []
        x = 0
        y = 0
        for row in self.matrix:
            for char in row:
                if char == " " :
                    robots.append((x, y,"@"))
                if char == ".":
                    robots.append((x, y,"+"))
                x = x + 1
            x = 0
            y = y + 1

        robots = random.sample(robots, self.n_robots)
        for robot in robots:
            self.set_content(robot[0], robot[1], robot[2])
        return [(robot[0], robot[1]) for robot in robots]

    def load_size(self):
        x = 0
        y = len(self.matrix)
        for row in self.matrix:
            if len(row) > x:
                x = len(row)
        return (x * 32, y * 32)

    def get_matrix(self):
        return self.matrix

    def print_matrix(self):
        for row in self.matrix:
            for char in row:
                sys.stdout.write(char)
                sys.stdout.flush()
            sys.stdout.write('\n')

    def get_content(self, x, y):
        return self.matrix[y][x]

    def set_content(self, x, y, content):
        if self.is_valid_value(content):
            self.matrix[y][x] = content
        else:
            print("ERROR: Value '"+content+"' to be added is not valid")

    def can_move(self, x, y):
        return self.get_content(self.robots[self.index][0]+x, self.robots[self.index][1]+y) not in ['#', '*', '$', '@']

    def next(self, x, y):
        return self.get_content(self.robots[self.index][0]+x, self.robots[self.index][1]+y)

    def can_push(self, x, y):
        return (self.next(x, y) in ['*', '$'] and self.next(x+x, y+y) in [' ', '.'])

    def is_completed(self):
        for row in self.matrix:
            for cell in row:
                if cell == '$':
                    return False
        return True

    def move_box(self, x, y, a, b):
        #        (x,y) -> move to do
        #        (a,b) -> box to move
        current_box = self.get_content(x, y)
        future_box = self.get_content(x+a, y+b)
        if current_box == '$' and future_box == ' ':
            self.set_content(x+a, y+b, '$')
            self.set_content(x, y, ' ')
        elif current_box == '$' and future_box == '.':
            self.set_content(x+a, y+b, '*')
            self.set_content(x, y, ' ')
        elif current_box == '*' and future_box == ' ':
            self.set_content(x+a, y+b, '$')
            self.set_content(x, y, '.')
        elif current_box == '*' and future_box == '.':
            self.set_content(x+a, y+b, '*')
            self.set_content(x, y, '.')

    # def unmove(self):
    #     if not self.queue.empty():
    #         movement = self.queue.get()
    #         if movement[2]:
    #             current = self.worker()
    #             self.move(movement[0] * -1, movement[1] * -1, False)
    #             self.move_box(current[0]+movement[0], current[1] +
    #                           movement[1], movement[0] * -1, movement[1] * -1)
    #         else:
    #             self.move(movement[0] * -1, movement[1] * -1, False)
    def action(self, choice):
        if choice == 0:
            moves = self.move(0, -1, True)
        elif choice == 1:
            moves = self.move(0, 1, True)
        elif choice == 2:
            moves = self.move(-1, 0, True)
        elif choice == 3:
            moves = self.move(1, 0, True)
        elif choice == 4:
            moves = [Move((0,0),(0,0))]
        else:
            print("choice not supported")
            print(choice)

        self.index += 1
        self.index = self.index % len(self.robots)
        return moves

    def move(self, x, y, save):
        moves = []
        if self.can_move(x, y):
            current = self.robots[self.index]
            char = self.get_content(current[0], current[1])
            future = self.next(x, y)
            moves.append(Move(
                (current[0], current[1]),
                (current[0]+x, current[1]+y)
            ))
            if char == '@' and future == ' ':
                # worker to floor
                self.set_content(current[0]+x, current[1]+y, '@')
                self.set_content(current[0], current[1], ' ')
                if save:
                    self.queue.put((x, y, False))
            elif char == '@' and future == '.':
                # worker to goal
                self.set_content(current[0]+x, current[1]+y, '+')
                self.set_content(current[0], current[1], ' ')
                if save:
                    self.queue.put((x, y, False))
            elif char == '+' and future == ' ':
                # worker on goal to floor
                self.set_content(current[0]+x, current[1]+y, '@')
                self.set_content(current[0], current[1], '.')
                if save:
                    self.queue.put((x, y, False))
            elif char == '+' and future == '.':
                # worker on goal to goal
                self.set_content(current[0]+x, current[1]+y, '+')
                self.set_content(current[0], current[1], '.')
                if save:
                    self.queue.put((x, y, False))
            self.robots[self.index] = (current[0]+x, current[1]+y)
        elif self.can_push(x, y):
            current = self.robots[self.index]
            char = self.get_content(current[0], current[1])
            future = self.next(x, y)
            future_box = self.next(x+x, y+y)
            moves.append(Move(
                (current[0], current[1]),
                (current[0]+x, current[1]+y)
            ))
            moves.append(Move(
                (current[0]+x, current[1]+y),
                (current[0]+2*x, current[1]+2*y)
            ))
            if char == '@' and future == '$' and future_box == ' ':
                # worker push box to floor
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], ' ')
                self.set_content(current[0]+x, current[1]+y, '@')
                if save:
                    self.queue.put((x, y, True))
            elif char == '@' and future == '$' and future_box == '.':
                # worker push box to goal
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], ' ')
                self.set_content(current[0]+x, current[1]+y, '@')
                if save:
                    self.queue.put((x, y, True))
            elif char == '@' and future == '*' and future_box == ' ':
                # worker push box on goal to floor
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], ' ')
                self.set_content(current[0]+x, current[1]+y, '+')
                if save:
                    self.queue.put((x, y, True))
            elif char == '@' and future == '*' and future_box == '.':
                # worker push box on goal to goal
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], ' ')
                self.set_content(current[0]+x, current[1]+y, '+')
                if save:
                    self.queue.put((x, y, True))
            if char == '+' and future == '$' and future_box == ' ':
                # worker on goal push box to floor
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], '.')
                self.set_content(current[0]+x, current[1]+y, '@')
                if save:
                    self.queue.put((x, y, True))
            elif char == '+' and future == '$' and future_box == '.':
                # worker on goal push box to goal
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], '.')
                self.set_content(current[0]+x, current[1]+y, '@')
                if save:
                    self.queue.put((x, y, True))
            elif char == '+' and future == '*' and future_box == ' ':
                # worker on goal push box on goal to floor
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], '.')
                self.set_content(current[0]+x, current[1]+y, '+')
                if save:
                    self.queue.put((x, y, True))
            elif char == '+' and future == '*' and future_box == '.':
                # worker on goal push box on goal to goal
                self.move_box(current[0]+x, current[1]+y, x, y)
                self.set_content(current[0], current[1], '.')
                self.set_content(current[0]+x, current[1]+y, '+')
                if save:
                    self.queue.put((x, y, True))
            self.robots[self.index] = (current[0]+x, current[1]+y)
        else:
            moves.append(Move(
                (0, 0),
                (0, 0)
            ))
        return moves

    def step(self, action):
        self.action(action)
        self.episode_step += 1
        if self.episode_step > 50:
            return self.get_state(), self.reward(), True
        else:
            return self.get_state(), self.reward(), self.is_completed()

    def get_state(self):
        state = np.zeros((10,11,1))
        x = 0
        y = 0
        for row in self.matrix:
            for char in row:
                state[y][x]=[self.convert[char]]
                x = x + 1
            x = 0
            y = y + 1
        state[self.robots[self.index][1],self.robots[self.index][0]] = [1]
        return state

    def reward(self):
        goal = []
        box = []
        x = 0
        y = 0
        for row in self.matrix:
            for char in row:
                if char == "." or char == "*" or char =="+":
                    goal = (x,y)
                if char == '$' or char == "*":
                    box = (x,y)
                x = x + 1
            x = 0
            y = y + 1
        if(self.is_completed()):
            return 50
        elif box[0] == goal[0] and box[1] == goal[1]:
            return 1
        else:
            return (1.0/math.sqrt(
                (box[0] - goal[0])**2 + 
                (box[1] - goal[1])**2
                ))
        


def print_game(matrix, screen):
    screen.fill(background)
    x = 0
    y = 0
    for row in matrix:
        for char in row:
            if char == ' ':  # floor
                screen.blit(floor, (x, y))
            elif char == '#':  # wall
                screen.blit(wall, (x, y))
            elif char == '@':  # worker on floor
                screen.blit(worker, (x, y))
            elif char == '.':  # dock
                screen.blit(docker, (x, y))
            elif char == '*':  # box on dock
                screen.blit(box_docked, (x, y))
            elif char == '$':  # box
                screen.blit(box, (x, y))
            elif char == '+':  # worker on dock
                screen.blit(worker_docked, (x, y))
            x = x + 32
        x = 0
        y = y + 32


def get_key():
  while 1:
    event = pygame.event.poll()
    if event.type == pygame.KEYDOWN:
      return event.key
    else:
      pass


def display_box(screen, message):
  "Print a message in a box in the middle of the screen"
  fontobject = pygame.font.Font(None, 18)
  pygame.draw.rect(screen, (0, 0, 0),
                   ((screen.get_width() / 2) - 100,
                    (screen.get_height() / 2) - 10,
                    200, 20), 0)
  pygame.draw.rect(screen, (255, 255, 255),
                   ((screen.get_width() / 2) - 102,
                    (screen.get_height() / 2) - 12,
                    204, 24), 1)
  if len(message) != 0:
    screen.blit(fontobject.render(message, 1, (255, 255, 255)),
                ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
  pygame.display.flip()


def display_end(screen):
    message = "Level Completed"
    fontobject = pygame.font.Font(None, 18)
    pygame.draw.rect(screen, (0, 0, 0),
                     ((screen.get_width() / 2) - 100,
                      (screen.get_height() / 2) - 10,
                      200, 20), 0)
    pygame.draw.rect(screen, (255, 255, 255),
                     ((screen.get_width() / 2) - 102,
                      (screen.get_height() / 2) - 12,
                      204, 24), 1)
    screen.blit(fontobject.render(message, 1, (255, 255, 255)),
                ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
    pygame.display.flip()


def ask(screen, question):
  "ask(screen, question) -> answer"
  pygame.font.init()
  current_string = []
  display_box(screen, question + ": " + "".join(current_string))
  while 1:
    inkey = get_key()
    if inkey == pygame.K_BACKSPACE:
      current_string = current_string[0:-1]
    elif inkey == pygame.K_RETURN:
      break
    elif inkey == pygame.K_MINUS:
      current_string.append("_")
    elif inkey <= 127:
      current_string.append(chr(inkey))
    display_box(screen, question + ": " + "".join(current_string))
  return "".join(current_string)


def start_game():
    start = pygame.display.set_mode((320, 240))
    level = ask(start, "Select Level")
    if int(level) > 0:
        return level
    else:
        print("ERROR: Invalid Level: "+str(level))
        sys.exit(2)

def checkSameBox(moves):
    for i, move in enumerate(moves):
        for move2 in moves[i:]:
            if(move.end[0] == move2.start[0] and move.end[1] == move2.start[1]):
                return True
    return False