import copy
import math
import random
from math import atan2

import pygame
import torch
from torch import nn, tensor

from drawable import Drawable


class Creature(Drawable):
    _default_reproduction_tick = 500
    _photosynthesis_scale = 0.01
    _maximum_age = 100001
    _game = None
    _eating_distance = 2

    def __init__(self, x: int, y: int, image: str = "images/blank.png", genome=None, net=None):
        super().__init__(x, y, image)
        # Physical attributes
        self.age: int = 0
        if genome is not None:
            self.genome: list = genome
        else:
            # 0 - speed
            # 1 - photosynthesis
            # 2 - reproduction: 0 - asexual; 1 - sexual
            # 3 - vision radius
            self.genome: list = [0] * 4  # vision radius
        self.recolor()
        if net is not None:
            self.net: Net = net
        else:
            self.net: Net = Net()
        self.debug: bool = False
        self.dead: bool = False
        self.action: int = -1
        self._actions = [
            {"action": self.up, "name": "up", "id": 0},
            {"action": self.down, "name": "down", "id": 1},
            {"action": self.left, "name": "left", "id": 2},
            {"action": self.right, "name": "right", "id": 3},
            {"action": self.eat, "name": "eat", "id": 4}
        ]

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
        if self.age > Creature._maximum_age:
            self.dead = True

    def is_dead(self):
        return self.dead

    def think(self):
        nearest_creature: Creature = self.find_nearest_creature()
        nearest_grass, nearest_distance = self.find_nearest_grass()
        distance_creature = float("inf")
        creature_radians = -1
        grass_radians = -1
        nearest_color = 0
        if nearest_creature is not None:
            nearest_color = nearest_creature.greenness() + nearest_creature.blueness()
            # angle
            delta_x = nearest_creature.x - self.x
            delta_y = nearest_creature.y - self.y
            creature_radians = atan2(delta_y, delta_x)
            # distance
            distance_creature = math.sqrt(
                delta_x * delta_x + delta_y * delta_y)
        if nearest_grass is not None:
            # angle
            delta_x = nearest_grass.x - self.x
            delta_y = nearest_grass.y - self.y
            grass_radians = atan2(delta_y, delta_x)

        self.action = self.net(
            tensor([creature_radians,
                    distance_creature,
                    nearest_color,
                    grass_radians,
                    nearest_distance
                    ])).tolist()

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
        self.image.fill(
            (0, self.greenness(), self.blueness(), 255),
            special_flags=pygame.BLEND_RGBA_ADD
        )

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

    def find_nearest_grass(self):
        grasses = Creature._game.grass
        min_distance = float("inf")
        min_grass = None
        for grass in grasses:
            dx = grass.x - self.x
            dy = grass.y - self.y
            distance = abs(dx) + abs(dy)
            if distance < min_distance:
                min_distance = distance
                min_grass = grass
        return min_grass, min_distance

    def act(self):
        if self.action >= 0:
            self._actions[self.action]["action"]()

    def up(self):
        self.y -= self.get_speed()
        self.y = max(0, self.y)

    def down(self):
        self.y += self.get_speed()
        self.y = min(self._game.height(), self.y)

    def left(self):
        self.x -= self.get_speed()
        self.x = max(0, self.x)

    def right(self):
        self.x += self.get_speed()
        self.x = min(self._game.width(), self.x)

    def eat(self):
        nearest_grass, nearest_distance = self.find_nearest_grass()
        if nearest_distance <= Creature._eating_distance:
            self.mass = nearest_grass.mass
            nearest_grass.kill()

    def is_clicked(self, x, y):
        return self.x < x < self.x + self.mass and self.y < y < self.y + self.mass


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.syg = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc(x)
        x = self.syg(x)
        x = self.softmax(x)
        return x.argmax(dim=0)

    def mutate(self):
        for param in self.parameters():
            param.data += torch.rand_like(param)
