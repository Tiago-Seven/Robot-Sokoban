import math
class Move:
  def __init__(self, start, end):
    self.start = start
    self.end = end

  def __str__(self):
        return "Move: start is (%s,%s), end is (%s,%s)" % (self.start[0], self.start[1], self.end[0], self.end[1])

def closest_distance(box,goals):
  distances = []
  for goal in goals:
      distances.append(math.sqrt(
          (box[0] - goal[0])**2 + 
          (box[1] - goal[1])**2
      ))
  return min(distances)