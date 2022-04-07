import random
import sys
import pygame
import torch
import torch.nn as nn


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
        # self.model = nn.Sequential(
        #     nn.Linear(768, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 4),
        #     nn.LogSoftmax(dim=1)
        # )
        # print(self.model(torch.randn(1,768)))

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

    def draw(self):
        # colors:
        photosynthesis = self.get_photosynthesis()
        greenness = photosynthesis * 255 if photosynthesis > 0 else 0
        if greenness > 255:
            greenness = 255
        image = self.im.copy()
        image = pygame.transform.scale(image, (self.mass, self.mass))
        image.fill((0, greenness, 0, 255), special_flags=pygame.BLEND_RGBA_ADD)
        self.screen.blit(image, (self.x, self.y))

    def logic(self, game):
        # Behaive
        self.reproduce(game)
        self.tick_age()
        # Photosynthesise
        self.photosynthesise()

    def photosynthesise(self):
        self.mass = self.mass + self.get_photosynthesis() * Creature._photosynthesis_scale

    def reproduce(self, game):
        if self.genome[2] is not None:
            if not self.is_sexual_reproduction():
                if self.reproduction_tick <= 0:
                    r_x = random.randint(-50, 50)
                    r_y = random.randint(-50, 50)
                    new_grass = Creature(self.x + r_x, self.y + r_y, game.screen)
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


class Game:
    def __init__(self):
        pygame.init()
        self._size = 1024, 768
        self._black = 0, 0, 0
        self.screen = pygame.display.set_mode(self._size)
        self.creatures: list = []
        self.init_game()

    def loop(self):
        clock = pygame.time.Clock()
        while ...:
            ms = clock.tick(60)
            self.screen.fill(self._black)
            self.logic()
            self.draw()
            pygame.display.flip()

    def logic(self):
        # remove dead creatures
        for creature in self.creatures:
            if creature.is_dead():
                self.creatures.remove(creature)
        for creature in self.creatures:
            creature.logic(self)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                sys.exit(0)

    def init_game(self):
        for i in range(5):
            x = random.randint(0, self.screen.get_width())
            y = random.randint(0, self.screen.get_height())
            new_grass = Creature(x, y, self.screen)
            new_grass.genome[1] = 1
            new_grass.genome[2] = 0
            self.creatures.append(new_grass)

    def draw_creatures(self):
        for creature in self.creatures:
            creature.draw()

    def draw(self):
        self.draw_creatures()

    def add_creature(self, creature):
        self.creatures.append(creature)


if __name__ == '__main__':
    game = Game()
    game.loop()
