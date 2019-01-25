#!/usr/bin/env python

import pygame
from pygame.locals import *  # noqa
import sys
import random
import numpy as np
import os

class FlappyBird:
    def __init__(self, display):
        if display:
            self.screen = pygame.display.set_mode((400, 708))
        else:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.display.set_mode((1, 1))
        self.bird = pygame.Rect(65, 50, 50, 50)
        self.background = pygame.image.load("assets/background.png").convert()
        self.birdSprites = [pygame.image.load("assets/1.png").convert_alpha(),
                            pygame.image.load("assets/2.png").convert_alpha(),
                            pygame.image.load("assets/dead.png")]
        self.wallUp = pygame.image.load("assets/bottom.png").convert_alpha()
        self.wallDown = pygame.image.load("assets/top.png").convert_alpha()
        self.gap = 200
        self.wallx = 400
        self.birdY = 350
        self.jump = 0
        self.jumpSpeed = 10
        self.gravity = 5
        self.dead = False
        self.sprite = 0
        self.counter = 0
        self.offset = random.randint(-110, 110)
        self.updateCounter = True
        self.display = display

    def updateWalls(self):
        self.wallx -= 2
        if self.wallx < -80:
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.updateCounter = True

        if self.wallx < 40 and self.updateCounter:
            self.counter += 1
            self.updateCounter = False

    def birdUpdate(self):
        if self.jump:
            self.jumpSpeed -= 1
            self.birdY -= self.jumpSpeed
            self.jump -= 1
        else:
            self.birdY += self.gravity
            self.gravity += 0.2
        self.bird[1] = self.birdY
        upRect = pygame.Rect(self.wallx,
                             360 + self.gap - self.offset + 10,
                             self.wallUp.get_width() - 10,
                             self.wallUp.get_height())
        downRect = pygame.Rect(self.wallx,
                               0 - self.gap - self.offset - 10,
                               self.wallDown.get_width() - 10,
                               self.wallDown.get_height())
        if upRect.colliderect(self.bird):
            self.dead = True
        if downRect.colliderect(self.bird):
            self.dead = True
        # The changes    
        if self.bird[1] > 708:
            self.dead = True
        elif self.bird[1] < 10:
            self.dead = True
        if not 0 < self.bird[1] < 720:
            self.bird[1] = 50
            self.birdY = 50
            self.dead = False
            self.counter = 0
            self.wallx = 400
            self.offset = random.randint(-110, 110)
            self.gravity = 5

    def jumpAction(self):
        self.jump = 17
        self.gravity = 5
        self.jumpSpeed = 10

    def step(self, clock, font):
        clock.tick(60)
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         sys.exit()
        #     if (event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN) and not self.dead:
        #         self.jump = 17
        #         self.gravity = 5
        #         self.jumpSpeed = 10

        self.screen.fill((255, 255, 255))
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.wallUp,
                            (self.wallx, 360 + self.gap - self.offset))
        self.screen.blit(self.wallDown,
                            (self.wallx, 0 - self.gap - self.offset))
        self.screen.blit(font.render(str(self.counter),
                                        -1,
                                        (255, 255, 255)),
                            (200, 50))
        if self.dead:
            self.sprite = 2
            return False, pygame.surfarray.array3d(pygame.display.get_surface())
        elif self.jump:
            self.sprite = 1
        self.screen.blit(self.birdSprites[self.sprite], (70, self.birdY))
        if not self.dead:
            self.sprite = 0
        self.updateWalls()
        self.birdUpdate()
        if self.display:
            pygame.display.update()
        return True, pygame.surfarray.array3d(pygame.display.get_surface())


if __name__ == '__main__':
    
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 50)
    clock = pygame.time.Clock()
    while FlappyBird(True).step(clock, font):
        pass