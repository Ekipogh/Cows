import random
import sys
import pygame
from PIL import Image
from torchvision import transforms
from creature import Creature


class Game:
    def __init__(self):
        pygame.init()
        self._size = 1024, 1024
        self._black = 0, 0, 0
        self.screen = pygame.display.set_mode(self._size)
        self.creatures: list = []
        self.init_game()

    def loop(self):
        while ...:
            self.logic()
            self.draw()

    def screenshot(self, x: int, y: int, radius: int):
        # make a screenshot
        left = x - radius
        top = y - radius
        width = radius * 2
        left = max(left, 0)
        top = max(left, 0)
        if left + width > self.screen.get_width():
            left = self.screen.get_width() - width
        if top + width > self.screen.get_height():
            top = self.screen.get_height() - width
        rect = pygame.Rect(left, top, width, width)
        sub = self.screen.subsurface(rect)
        raw_str = pygame.image.tostring(sub, "RGB", False)
        screenshot = Image.frombytes("RGB", (width, width), raw_str)
        return transforms.ToTensor()(screenshot)

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
        for _ in range(5):
            x = random.randint(0, self.screen.get_width())
            y = random.randint(0, self.screen.get_height())
            new_grass = Creature(x, y)
            new_grass.set_photosynthesis(True)
            new_grass.set_sexual_reproduction(False)
            self.add_creature(new_grass)
        x = random.randint(0, self.screen.get_width())
        y = random.randint(0, self.screen.get_height())
        new_cow = Creature(x, y)
        new_cow.mass = 50
        new_cow.set_photosynthesis(False)
        new_cow.set_sexual_reproduction(True)
        self.add_creature(new_cow)

    def draw_creatures(self):
        for creature_to_draw in self.creatures:
            creature_to_draw.draw(self)

    def draw(self):
        self.screen.fill(self._black)
        self.draw_creatures()
        pygame.display.flip()

    def add_creature(self, creature_to_add):
        self.creatures.append(creature_to_add)
