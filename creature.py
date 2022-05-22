import random
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import cows


class Creature:
    _default_reproduction_tick = 1000
    _photosynthesis_scale = 0.01
    _maximum_age = 2001

    def __init__(self, x: int, y: int, screen: pygame.Surface, image: str = "images/blank.png"):
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

    def logic(self, game_state: cows.Game):
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

    def reproduce(self, game: cows.Game):
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

    def think(self, vision):
        foobar = self.net(vision)
        print(foobar)

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
