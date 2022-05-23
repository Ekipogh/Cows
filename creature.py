import random
import pygame
import torch
from torch import nn
import torch.nn.functional as F


class Creature:
    _default_reproduction_tick = 1000
    _photosynthesis_scale = 0.01
    _maximum_age = 2001

    def __init__(self, x: int, y: int, image: str = "images/blank.png"):
        self.x = x
        self.y = y
        self.image: pygame.Surface = pygame.image.load(image)
        rect = self.image.get_rect()
        rect.move(self.x, self.y)
        # Physical attributes
        self.age = 0
        self.mass = 0
        # 0 - speed
        # 1 - photosynthesis
        # 2 - reproduction: 0 - asexual; 1 - sexual
        # 3 - vision radius
        self.genome = [0] * 4
        self.genome[3] = 16
        self.net = Net()

    def get_speed(self):
        return self.genome[0]

    def get_photosynthesis(self):
        return self.genome[1]

    def set_photosynthesis(self, photosynthesis: int):
        self.genome[1] = photosynthesis

    def set_sexual_reproduction(self, reproduction: int):
        self.genome[2] = reproduction

    def draw(self, game_state):
        # colors:
        photosynthesis = self.get_photosynthesis()
        greenness = photosynthesis * 255 if photosynthesis > 0 else 0
        blueness = 255 if photosynthesis == 0 else 0
        greenness = min(greenness, 255)
        image = self.image.copy()
        image = pygame.transform.scale(image, (self.mass, self.mass))
        image.fill((0, greenness, blueness, 255), special_flags=pygame.BLEND_RGBA_ADD)
        game_state.screen.blit(image, (self.x, self.y))

    def logic(self, game_state):
        # Behave
        self.reproduce(game_state)
        self.tick_age()
        # Photosynthesise
        self.photosynthesise()
        # Get a screenshot
        screenshot = game_state.screenshot(self.x, self.y, self.get_vision())
        # Think and act
        self.think(screenshot)

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
                    new_grass = Creature(new_x, new_y)
                    new_grass.genome = self.genome.copy()
                    game_state.add_creature(new_grass)

    def is_sexual_reproduction(self):
        return self.genome[2] == 1

    def tick_age(self):
        if self.age >= Creature._maximum_age:
            return
        self.age += 1

    def is_dead(self):
        return self.age > self._maximum_age

    def think(self, vision):
        pass

    def get_vision(self):
        return self.genome[3]

    def set_vision(self, vision: int):
        self.genome[3] = vision


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
