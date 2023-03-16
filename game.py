import random
import pygame
from PIL import Image
from torchvision import transforms
from creature import Creature
from grass import Grass


class Game:
    def __init__(self):
        self.running = True
        pygame.init()
        self._size = 1024, 1024
        self._black = 0, 0, 0
        self.screen = pygame.display.set_mode(self._size)
        self.creatures: list = []
        self.grass: list = []
        self.init_game()

    def loop(self):
        while self.running:
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
        creature: Creature
        for creature in self.creatures:
            if creature.is_dead():
                self.creatures.remove(creature)
        # remove dead grass
        grass: Grass
        for grass in self.grass:
            if grass.is_dead():
                self.grass.remove(grass)
        for creature in self.creatures:
            creature.logic(self)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False

    def init_game(self):
        Creature.set_game(self)
        for _ in range(5):
            x = random.randint(0, self.screen.get_width())
            y = random.randint(0, self.screen.get_height())
            new_grass = Grass(x, y)
            self.add_grass(new_grass)
        for _ in range(5):
            x = random.randint(0, self.screen.get_width())
            y = random.randint(0, self.screen.get_height())
            new_cow = Creature(x, y)
            new_cow.mass = 50
            new_cow.set_photosynthesis(False)
            new_cow.set_sexual_reproduction(False)
            new_cow.set_speed(0.02)
            new_cow.set_vision(15)
            if _ == 0:
                new_cow.debug = True
            self.add_creature(new_cow)

    def draw_creatures(self):
        for creature_to_draw in self.creatures:
            creature_to_draw.draw(self)

    def draw(self):
        self.screen.fill(self._black)
        self.draw_creatures()
        self.draw_grass()
        pygame.display.flip()

    def add_creature(self, creature_to_add):
        self.creatures.append(creature_to_add)

    def width(self):
        return self._size[0]

    def height(self):
        return self._size[1]

    def add_grass(self, grass):
        self.grass.append(grass)

    def draw_grass(self):
        for grass in self.grass:
            grass.draw(self)
