import copy
import random
from math import atan2

import pygame
import torch
from torch import nn, tensor


class Creature:
    _default_reproduction_tick = 500
    _photosynthesis_scale = 0.01
    _maximum_age = 100001
    _game = None

    def __init__(self, x: int, y: int, image: str = "images/blank.png", genome=None, net=None):
        self.actions = []
        self.x = x
        self.y = y
        self.image: pygame.Surface = pygame.image.load(image)
        rect = self.image.get_rect()
        rect.move(self.x, self.y)
        # Physical attributes
        self.age = 0
        self.mass = 1
        if genome is not None:
            self.genome = genome
        else:
            # 0 - speed
            # 1 - photosynthesis
            # 2 - reproduction: 0 - asexual; 1 - sexual
            # 3 - vision radius
            self.genome = [0] * 4  # vision radius
        self.recolor()
        if net is not None:
            self.net = net
        else:
            self.net = Net()

    def get_speed(self):
        return self.genome[0]

    def set_speed(self, speed):
        self.genome[0] = speed

    def get_photosynthesis(self):
        return self.genome[1]

    def set_photosynthesis(self, photosynthesis: int):
        self.genome[1] = photosynthesis
        self.recolor()

    def set_sexual_reproduction(self, reproduction: int):
        self.genome[2] = reproduction

    def draw(self, game_state):
        self.image = pygame.transform.scale(self.image, (self.mass, self.mass))
        game_state.screen.blit(self.image, (self.x, self.y))

    def logic(self, game_state):
        # Behave
        self.reproduce(game_state)
        self.tick_age()
        # Photosynthesise
        self.photosynthesise()
        # Think and act
        self.think()
        self.act()

    def photosynthesise(self):
        self.mass = self.mass + self.get_photosynthesis() * Creature._photosynthesis_scale

    def reproduce(self, game_state):
        if self.genome[2] is not None:
            if not self.is_sexual_reproduction():
                if self.age != 0 and self.age % Creature._default_reproduction_tick == 0:
                    r_x = random.randint(-50, 50)
                    r_y = random.randint(-50, 50)
                    new_x = self.x + r_x
                    new_x = min(new_x, game_state.screen.get_width() - 1)
                    new_x = max(new_x, 1)
                    new_y = self.y + r_y
                    new_y = min(new_y, game_state.screen.get_height() - 1)
                    new_y = max(new_y, 1)
                    new_creature = Creature(new_x, new_y, genome=self.genome.copy(),
                                            net=copy.deepcopy(self.net).mutate())

                    game_state.add_creature(new_creature)

    def is_sexual_reproduction(self):
        return self.genome[2] == 1

    def tick_age(self):
        self.age += 1

    def is_dead(self):
        return self.age > self._maximum_age

    def think(self):
        nearest = self.find_nearest_creature()
        if nearest is not None:
            # angle
            delta_x = nearest.x - self.x
            delta_y = nearest.y - self.y
            theta_radians = atan2(delta_y, delta_x)
            # distance
            distance = abs(delta_x) + abs(delta_y)
            self.actions = self.net(tensor([theta_radians, distance])).tolist()

    def get_vision(self):
        return self.genome[3]

    def set_vision(self, vision: int):
        self.genome[3] = vision

    def greenness(self):
        photosynthesis = self.get_photosynthesis()
        greenness = photosynthesis * 255 if photosynthesis > 0 else 0
        greenness = min(greenness, 255)
        return greenness

    def blueness(self):
        blueness = 255 if self.get_photosynthesis() == 0 else 0
        return blueness

    def recolor(self):
        self.image.fill((0, self.greenness(), self.blueness(), 255), special_flags=pygame.BLEND_RGBA_ADD)

    @classmethod
    def set_game(cls, game):
        if Creature._game is None:
            Creature._game = game

    def find_nearest_creature(self):
        creatures = Creature._game.creatures
        min_distance = float("inf")
        min_creature = None
        for creature in creatures:
            if creature is self:
                continue
            dx = creature.x - self.x
            dy = creature.y - self.y
            distance = abs(dx) + abs(dy)
            if distance < min_distance:
                min_distance = distance
                min_creature = creature
        return min_creature

    def act(self):
        up = self.actions[0]
        down = self.actions[1]
        left = self.actions[2]
        right = self.actions[3]
        self.move(up, down, left, right)

    def move(self, up, down, left, right):
        print(f"Up: {up} Down: {down} Left: {left} Right: {right}")
        self.x += right * self.get_speed()
        self.x -= left * self.get_speed()
        self.y += up * self.get_speed()
        self.y -= down * self.get_speed()
        self.x = min(self.x, self._game.width())
        self.y = min(self.y, self._game.height())
        self.x = max(0, self.x)
        self.y = max(0, self.y)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

    def mutate(self):
        for param in self.parameters():
            param.data += torch.rand_like(param)
