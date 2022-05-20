import random

import pygame
from torch import nn as nn


class Creature:
    _default_reproduction_tick = 1000
    _photosynthesis_scale = 0.01
    _maximum_age = 2001

    def __init__(self, x, y, screen, image: str = "images/blank.png"):
        self.screen = screen
        self.x = x
        self.y = y
        self.im: pygame.Surface = pygame.image.load(image)
        self.rect = self.im.get_rect()
        self.rect.move(self.x, self.y)
        # Physical attributes
        self.dead = False
        self.age = 0
        self.mass = 0
        self.reproduction_tick = Creature._default_reproduction_tick
        # 0 - speed
        # 1 - photosynthesis
        # 2 - reproduction: 0 - asexual; 1 - sexual
        self.genome = [None] * 3
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 6)
        )

    def get_speed(self):
        try:
            return self.genome[0] if self.genome[0] is not None else 0
        except IndexError:
            return 0

    def get_photosynthesis(self):
        try:
            return self.genome[1] if self.genome[1] is not None else 0
        except IndexError:
            return 0

    def set_photosynthesis(self, photosynthesis):
        self.genome[1] = int(photosynthesis)

    def set_sexual_reproduction(self, reproduction):
        self.genome[2] = int(reproduction)

    def draw(self):
        # colors:
        photosynthesis = self.get_photosynthesis()
        greenness = photosynthesis * 255 if photosynthesis > 0 else 0
        blueness = 255 if photosynthesis == 0 else 0
        if greenness > 255:
            greenness = 255
        image = self.im.copy()
        image = pygame.transform.scale(image, (self.mass, self.mass))
        image.fill((0, greenness, blueness, 255), special_flags=pygame.BLEND_RGBA_ADD)
        self.screen.blit(image, (self.x, self.y))

    def logic(self, game_state):
        # Behave
        self.reproduce(game_state)
        self.tick_age()
        # Photosynthesise
        self.photosynthesise()
        # Think and act
        self.think(game_state)

    def photosynthesise(self):
        self.mass = self.mass + self.get_photosynthesis() * Creature._photosynthesis_scale

    def reproduce(self, game):
        if self.genome[2] is not None:
            if not self.is_sexual_reproduction():
                if self.reproduction_tick <= 0:
                    r_x = random.randint(-50, 50)
                    r_y = random.randint(-50, 50)
                    new_grass = Creature(
                        self.x + r_x, self.y + r_y, game.screen)
                    new_grass.genome = self.genome.copy()
                    game.add_creature(new_grass)
                    self.reset_reproduction_tick()
                self.reproduction_tick -= 1

    def is_sexual_reproduction(self):
        return self.genome[2] == 1

    def reset_reproduction_tick(self):
        self.reproduction_tick = Creature._default_reproduction_tick

    def tick_age(self):
        if self.age >= Creature._maximum_age:
            self.dead = True
            return
        self.age += 1

    def is_dead(self):
        return self.dead

    def think(self, game_state):
        pass
