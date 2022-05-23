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
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        left = x - radius
        top = y - radius
        width = radius * 2
        height = radius * 2
        left = max(left, 0)
        top = max(top, 0)
        left = min(left, screen_width)
        top = min(top, screen_height)
        # if square is over right side
        if left + width > screen_width:
            width = screen_width - left
        if left < 0:
            width = width - left
        if top + height > screen_height:
            height = screen_height - top
        if top < 0:
            height = height - top
        rect = pygame.Rect(left, top, width, height)
        sub = self.screen.subsurface(rect)
        raw_str = pygame.image.tostring(sub, "RGB", False)
        screenshot = Image.frombytes("RGB", (width, height), raw_str)
        screenshot = Game.make_square(screenshot, min_size=(radius * 2))
        return transforms.ToTensor()(screenshot)

    @staticmethod
    def make_square(image: Image, min_size=256, fill_color=(255, 255, 255, 0)):
        x, y = image.size
        size = max(min_size, x, y)
        new_im = Image.new('RGBA', (size, size), fill_color)
        new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

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
