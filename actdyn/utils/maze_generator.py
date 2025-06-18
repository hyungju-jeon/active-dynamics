import random
import os
from datetime import datetime

SAVE_MODE = True

WIDTH = 39          # maze width (must be odd)
HEIGHT = 19         # maze height (must be odd)
assert WIDTH % 2 == 1 and WIDTH >= 3
assert HEIGHT % 2 == 1 and HEIGHT >= 3
SEED = 1
random.seed(SEED)

# characters for displaying the maze
EMPTY = ' '
MARK = '@'
WALL = chr(9608) # character 9608 is '█'
NORTH, SOUTH, EAST, WEST = 'n', 's', 'e', 'w'

# create filled-in maze data structure to start
maze = {}
for x in range(WIDTH):
    for y in range(HEIGHT):
        maze[(x, y)] = WALL # every space is a wall at first

def printMaze(maze, markX=None, markY=None, file=None):
    """displays maze data structure in the maze argument;
    the markX and markY arguments are coordinates of the current
    '@' location of the algorithm as it generates the maze
    If file is provided, write to file instead of stdout."""

    for y in range(HEIGHT):
        line = ''
        for x in range(WIDTH):
            if markX == x and markY == y:
                # display '@' mark here
                line += MARK
            else:
                # display wall or empty space
                line += maze[(x, y)]
        if file:
            file.write(line + '\n')
        else:
            print(line)


def visit(x, y):
    """carve out empty spaces in the maze at x, y and then
    recursively move to neighboring unvisited spaces;
    this function backtracks when the mark has reached a dead end"""
    maze[(x, y)] = EMPTY # carve out the space at x, y

    while True:
        # check which neighboring spaces adjacent to
        # the mark have not been visited already
        unvisitedNeighbors = []
        if y > 1 and (x, y - 2) not in hasVisited:
            unvisitedNeighbors.append(NORTH)

        if y < HEIGHT - 2 and (x, y + 2) not in hasVisited:
            unvisitedNeighbors.append(SOUTH)

        if x > 1 and (x - 2, y) not in hasVisited:
            unvisitedNeighbors.append(WEST)

        if x < WIDTH - 2 and (x + 2, y) not in hasVisited:
            unvisitedNeighbors.append(EAST)

        if len(unvisitedNeighbors) == 0:
            # BASE CASE
            # all neighboring spaces have been visited, so this is a dead end
            # backtrack to an earlier space
            return
        else:
            # RECURSIVE CASE
            # randomly pick an unvisited neighbor to visit
            nextIntersection = random.choice(unvisitedNeighbors)

            # move the mark to an unvisited neighboring space

            if nextIntersection == NORTH:
                nextX = x
                nextY = y - 2
                maze[(x, y - 1)] = EMPTY # connecting hallway
            elif nextIntersection == SOUTH:
                nextX = x
                nextY = y + 2
                maze[(x, y + 1)] = EMPTY # connecting hallway
            elif nextIntersection == WEST:
                nextX = x - 2
                nextY = y
                maze[(x - 1, y)] = EMPTY # connecting hallway
            elif nextIntersection == EAST:
                nextX = x + 2
                nextY = y
                maze[(x + 1, y)] = EMPTY # connecting hallway

            hasVisited.append((nextX, nextY)) # mark as visited
            visit(nextX, nextY) # recursively visit this space


# carve out paths in maze data structure
hasVisited = [(1, 1)] # start by visiting top-left corner
visit(1, 1)

# display final resulting maze data structure
if SAVE_MODE:
    # ensure the output directory exists
    output_dir = 'others/generated_mazes'
    os.makedirs(output_dir, exist_ok=True)

    # create a unique filename with timestamp and seed
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"maze_{WIDTH}x{HEIGHT}_seed{SEED}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)

    # write maze to file
    with open(filepath, 'w', encoding='utf-8') as f:
        printMaze(maze, file=f)
    print(f"Maze saved to {filepath}")

else:
    printMaze(maze)