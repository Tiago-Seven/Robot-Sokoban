import os, sys, math, pygame, pygame.mixer
from pygame.locals import *

black = 0, 0, 0
white = 255, 255, 255
background = 255, 226, 191
red = 255, 0, 0

wall = pygame.image.load('images/wall.png')
floor = pygame.image.load('images/floor.png')
box = pygame.image.load('images/box.png')
box_docked = pygame.image.load('images/box_docked.png')
worker = pygame.image.load('tiny_robot.png')
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
                    screen.blit(box_docked, (x, y))
                elif char == '$':  # box
                    screen.blit(box, (x, y))
                x = x + 32
            x = 0
            y = y + 32

class Real_Time_Display:
    def __init__(self, basic_map):
        self.basic_map = basic_map
  
    def run(self):
        size = self.basic_map.load_size()
        # pygame.init()
        screen = pygame.display.set_mode(size)
        clock = pygame.time.Clock()
        pygame.display.set_caption("First Class!")

        fps_limit = 30
        run_me = True
        robot = My_Robot(self.basic_map.load_size()[0]/2, self.basic_map.load_size()[1]/2)
        n = 0
        while n < 60 and run_me:
            n += 1
            clock.tick(fps_limit)
            self.basic_map.print_game(screen)

            robot.display(screen)
            robot.rotation += 10
            robot.x += 1
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit(0)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit(0)

class My_Robot:
    def __init__(self,x ,y, rotation = 0):
        self.x = x
        self.y = y
        self.rotation = rotation
        self.image = pygame.image.load("tiny_robot.png").convert_alpha()
        self.w, self.h = self.image.get_size()
  
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
        origin = (self.x - self.w/2 + min_box[0] - pivot_move[0],
                self.y - self.h/2 - max_box[1] + pivot_move[1])

        # get a rotated image
        rotated_image = pygame.transform.rotate(self.image, self.rotation)

        # rotate and blit the image
        surf.blit(rotated_image, origin)

