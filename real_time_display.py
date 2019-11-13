import os, sys, math, pygame, pygame.mixer
from pygame.locals import *

black = 0, 0, 0
white = 255, 255, 255
red = 255, 0, 0

class My_Robot:
  def __init__(self,x ,y, rotation = 0):
    self.x = x
    self.y = y
    self.rotation = rotation
    self.image = pygame.image.load("robot.png").convert_alpha()
    self.w, self.h = self.image.get_size()

  def display(self, surface):
    # this rect determines the position the ball is drawn
    # rot_image = pygame.transform.rotate(self.image, self.rotation)
    # surface.blit(rot_image, (self.x, self.y))

    rotated_image = pygame.transform.rotate(image, self.rotation)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)

    surf.blit(rotated_image, new_rect.topleft)
  
  def blitRotate(self,surf):
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

screen_size = screen_width, screen_height = 600, 400

screen = pygame.display.set_mode(screen_size)

clock = pygame.time.Clock()

pygame.display.set_caption("First Class!")

fps_limit = 30
run_me = True
robot = My_Robot(300, 200)
while run_me:
  clock.tick(fps_limit)

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      run_me = False
  
  screen.fill(white)
  robot.blitRotate(screen)
  robot.rotation += 10
  robot.x += 1
  pygame.display.flip()

pygame.quit()
sys.exit()
