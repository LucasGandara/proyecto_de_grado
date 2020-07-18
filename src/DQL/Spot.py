import pygame

cols = 32
rows = 32

class Spot(object):
    def __init__(self, x, y):
        self.f = 0
        self.g = 0
        self.h = 0
        self.x = x # posicion x del punto en el espacio
        self.y = y # Posicion y del punto en el espacio
        self.Neighbors = []
        self.Neighbors2 = []
        self.Neighbors3 = []
        self.previous = None
        self.obstacle = False
        self.color = pygame.Color(255, 255, 102)
        
    def addNeighbors(self, spots, n=1):
        if n == 1:
            if self.x >= 1:
                self.Neighbors.append(spots[self.x - 1][self.y])
            if self.x < (cols - 1):
                self.Neighbors.append(spots[self.x + 1][self.y])
            if self.y >= 1:
                self.Neighbors.append(spots[self.x][self.y - 1])
            if self.y < (rows - 1):
                self.Neighbors.append(spots[self.x][self.y + 1])
            if self.x > 0 and self.y > 0:
                self.Neighbors.append(spots[self.x - 1][self.y - 1])
            if self.x < cols - 1 and self.y > 0:
                self.Neighbors.append(spots[self.x + 1][self.y - 1])
            if self.x > 0 and self.y < rows - 1:
                self.Neighbors.append(spots[self.x - 1][self.y + 1])
            if self.x < cols - 1 and self.y < rows - 1:
                self.Neighbors.append(spots[self.x + 1][self.y + 1])

        if n == 2:
            if self.x >= 2:
                self.Neighbors2.append(spots[self.x - 2][self.y])
            if self.x < (cols - 2):
                self.Neighbors2.append(spots[self.x + 2][self.y])
            if self.y >= 2:
                self.Neighbors2.append(spots[self.x][self.y - 2])
            if self.y < (rows - 2):
                self.Neighbors2.append(spots[self.x][self.y + 2])
            if self.x > 2 and self.y > 2:
                self.Neighbors2.append(spots[self.x - 1][self.y - 2])
                self.Neighbors2.append(spots[self.x - 2][self.y - 1])
                self.Neighbors2.append(spots[self.x - 2][self.y - 2])
            if self.x < cols - 2 and self.y > 1:
                self.Neighbors2.append(spots[self.x + 2][self.y - 1])
                self.Neighbors2.append(spots[self.x + 2][self.y - 2])
                self.Neighbors2.append(spots[self.x + 1][self.y - 2])
            if self.x > 1 and self.y < rows - 2:
                self.Neighbors2.append(spots[self.x - 1][self.y + 2])
                self.Neighbors2.append(spots[self.x - 2][self.y + 1])
                self.Neighbors2.append(spots[self.x - 2][self.y + 2])
            if self.x < cols - 2 and self.y < rows - 2:
                self.Neighbors2.append(spots[self.x + 1][self.y + 2])
                self.Neighbors2.append(spots[self.x + 2][self.y + 2])
                self.Neighbors2.append(spots[self.x + 2][self.y + 1])

        if n == 3:
            if self.x >= 2:
                self.Neighbors3.append(spots[self.x - 3][self.y])
            if self.x < (cols - 3):
                self.Neighbors3.append(spots[self.x + 3][self.y])
            if self.y >= 2:
                self.Neighbors3.append(spots[self.x][self.y - 3])
            if self.y < (rows - 2):
                self.Neighbors3.append(spots[self.x][self.y + 3])
            if self.x > 3 and self.y > 3:
                self.Neighbors3.append(spots[self.x - 1][self.y - 3])
                self.Neighbors3.append(spots[self.x - 3][self.y - 1])
                self.Neighbors3.append(spots[self.x - 3][self.y - 2])
                self.Neighbors3.append(spots[self.x - 2][self.y - 3])
                self.Neighbors3.append(spots[self.x - 3][self.y - 3])
            if self.x < cols - 3 and self.y > 2:
                self.Neighbors3.append(spots[self.x + 3][self.y - 1])
                self.Neighbors3.append(spots[self.x + 3][self.y - 2])
                self.Neighbors3.append(spots[self.x + 1][self.y - 3])
                self.Neighbors3.append(spots[self.x + 2][self.y - 3])
                self.Neighbors3.append(spots[self.x + 3][self.y - 3])
            if self.x > 3 and self.y < rows - 3:
                self.Neighbors3.append(spots[self.x - 1][self.y + 3])
                self.Neighbors3.append(spots[self.x - 3][self.y + 1])
                self.Neighbors3.append(spots[self.x - 3][self.y + 2])
                self.Neighbors3.append(spots[self.x - 2][self.y + 3])
                self.Neighbors3.append(spots[self.x - 3][self.y + 3])
            if self.x < cols - 3 and self.y < rows - 3:
                self.Neighbors3.append(spots[self.x + 1][self.y + 3])
                self.Neighbors3.append(spots[self.x + 2][self.y + 3])
                self.Neighbors3.append(spots[self.x + 3][self.y + 1])
                self.Neighbors3.append(spots[self.x + 3][self.y + 2])
                self.Neighbors3.append(spots[self.x + 3][self.y + 3])

    def get_list_of_neigbors(self, n=1):
        """ Return the list of the neighbors on the given 
            index """
        
        Neighbors = []
        if n == 1:
            for neighbor in self.Neighbors:
                Neighbors.append([neighbor.x, neighbor.y])
        elif n == 2:
            for neighbor in self.Neighbors2:
                Neighbors.append([neighbor.x, neighbor.y])
        elif n == 3:
            for neighbor in self.Neighbors3:
                Neighbors.append([neighbor.x, neighbor.y])

        return Neighbors
