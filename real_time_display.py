import os, sys, math, pygame, pygame.mixer
from pygame.locals import *
from utils import Move

black = 0, 0, 0
white = 255, 255, 255
background = 255, 226, 191
red = 255, 0, 0

wall = pygame.image.load('images/wall.png')
floor = pygame.image.load('images/floor.png')
box = pygame.image.load('images/box.png')
box_docked = pygame.image.load('images/box_docked.png')
worker_docked = pygame.image.load('images/worker_dock.png')
docker = pygame.image.load('images/dock.png')

class Basic_Map:
    def __init__(self, matrix):
        self.matrix = matrix
  
    def load_size(self):
        x = 0
        y = len(self.matrix)
        for row in self.matrix:
          if len(row) > x:
            x = len(row)
        return (x * 32, y * 32)

    def print_game(self, screen):
        screen.fill(background)
        x = 0
        y = 0
        for row in self.matrix:
            for char in row:
                if char == ' ':  # floor
                    screen.blit(floor, (x, y))
                elif char == '#':  # wall
                    screen.blit(wall, (x, y))
                elif char == '.':  # dock
                    screen.blit(docker, (x, y))
                elif char == '*':  # box on dock
                    screen.blit(docker, (x, y))
                elif char == '+':  # worker on dock
                    screen.blit(docker, (x, y))
                x = x + 32
            x = 0
            y = y + 32

class Real_Time_Display:
    def __init__(self, basic_map):
        self.basic_map = basic_map
        self.boxes = []
        self.robots = []
        x = 0
        y = 0
        for row in self.basic_map.matrix:
            for char in row:
                if char == '@':  # worker on floor
                    self.robots.append(Robot((x,y)))
                elif char == '*':  # box on dock
                    self.boxes.append(Box((x, y), True))
                elif char == '$':  # box
                    self.boxes.append(Box((x, y), False))
                elif char == '+':  # worker on dock
                    self.robots.append(Robot((x, y)))
                x = x + 32
            x = 0
            y = y + 32

  
    def run(self,moves):
        size = self.basic_map.load_size()
        # pygame.init()
        screen = pygame.display.set_mode(size)
        clock = pygame.time.Clock()
        pygame.display.set_caption("First Class!")

        fps_limit = 30
        run_me = True
        n = 0

        for object in self.robots + self.boxes:
            for move in moves:
                if object.x == move.start[0] * 32 and object.y == move.start[1] * 32:
                    object.move = (move.end[0] - move.start[0],
                                   move.end[1] - move.start[1])
                    object.dest = (move.end[0]*32, move.end[1]*32)
                    if isinstance(object,Box):
                        object.is_in_goal = self.basic_map.matrix[move.end[1]][move.end[0]] == '*'
        
        while n < 60 and run_me:
            n += 1
            clock.tick(fps_limit)
            self.basic_map.print_game(screen)

            all_rotated = True
            for robot in self.robots:
                if not robot.is_rotated():
                    all_rotated = False
                    break
            
            if not all_rotated:
                for robot in self.robots:
                    robot.update_rotate()
                    robot.display(screen)
                for box in self.boxes:
                    box.display(screen)
            else:
                for object in self.boxes + self.robots:
                    object.update_move()
                    object.display(screen)
                
                all_placed = True
                for object in self.robots + self.boxes:
                    if not object.is_in_place():
                        all_placed = False
                        break

                run_me = not all_placed

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit(0)
            


class Robot:
    def __init__(self,pos, rotation = 0):
        self.x = pos[0]
        self.y = pos[1]
        self.move = (0, 0)
        self.dest = pos
        self.rotation = rotation
        self.image = pygame.image.load("images/tiny_robot.png").convert_alpha()
        self.w, self.h = self.image.get_size()
    
    def is_in_place(self):
        return self.x == self.dest[0] and self.y == self.dest[1]

    def is_rotated(self):
        if self.move == (0,1):
            return self.rotation % 360 == 0
        elif self.move == (0,-1):
            return self.rotation % 360 == 180
        elif self.move == (1,0):
            return self.rotation % 360 == 90
        elif self.move == (-1,0):
            return self.rotation % 360 == 270
        else:
            return True

    def update_move(self):
        if not self.is_in_place():
            self.x += self.move[0]
            self.y += self.move[1]

    def update_rotate(self):
        if not self.is_rotated():
            if self.move == (0, 1):
                if self.rotation < 180:
                    self.rotation -= 10
                else:
                    self.rotation += 10
            elif self.move == (0,-1):
                if self.rotation < 180:
                    self.rotation += 10
                else:
                    self.rotation -= 10
            elif self.move == (1,0):
                if 90 < self.rotation < 270:
                    self.rotation -= 10
                else:
                    self.rotation += 10
            elif self.move == (-1,0):
                if 90 < self.rotation < 270:
                    self.rotation += 10
                else:
                    self.rotation -= 10
            self.rotation = self.rotation % 360
            if self.rotation < 0:
                self.rotation += 360

    def display(self,surf):
        # calcaulate the axis aligned bounding box of the rotated image
        w, h = self.image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(self.rotation) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[
                0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[
                0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        pivot = pygame.math.Vector2(self.w/2, -self.h/2)
        pivot_rotate = pivot.rotate(self.rotation)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (self.x+16 - self.w/2 + min_box[0] - pivot_move[0],
                self.y+16 - self.h/2 - max_box[1] + pivot_move[1])

        # get a rotated image
        rotated_image = pygame.transform.rotate(self.image, self.rotation)

        # rotate and blit the image
        surf.blit(rotated_image, origin)

class Box:
    def __init__(self, pos, is_in_goal):
        self.x = pos[0]
        self.y = pos[1]
        self.move = (0, 0)
        self.dest = pos
        self.rotation = 0
        self.image = pygame.image.load("images/box.png").convert_alpha()
        self.image_goal = pygame.image.load("images/box_docked.png").convert_alpha()
        self.w, self.h = self.image.get_size()
        self.is_in_goal = is_in_goal

    def is_in_place(self):
        return self.x == self.dest[0] and self.y == self.dest[1]

    def update_move(self):
        if not self.is_in_place():
            self.x += self.move[0]
            self.y += self.move[1]

    def display(self, surf):
        # rotate and blit the image
        if self.is_in_goal and self.is_in_place():
            surf.blit(self.image_goal, (self.x, self.y))
        else:
            surf.blit(self.image, (self.x, self.y))
