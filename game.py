import random
import pygame
from PIL import Image
from torchvision import transforms
from creature import Creature
from grass import Grass


class Game:
    _MAX_GRASS = 2000
    _GRASS_SPAWN_AMOUNT = 100
    _CREATURE_SPAWN_AMOUNT = 6
    
    def __init__(self):
        self.running = True
        pygame.init()
        self._size = 1024, 1024
        self._black = 0, 0, 0
        self.screen = pygame.display.set_mode(self._size)
        self.creatures: list = []
        self.grass: list = []
        self.dead_creatures: list = []
        self.debug_creature: Creature = None
        self.generation = 0
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
        screenshot = Game.make_square(screenshot, min_size=radius * 2)
        return transforms.ToTensor()(screenshot)

    @staticmethod
    def make_square(image: Image, min_size=256, fill_color=(255, 255, 255, 0)):
        x, y = image.size
        size = max(min_size, x, y)
        new_im = Image.new('RGBA', (size, size), fill_color)
        new_im.paste(image, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def logic(self):
        for creature in self.creatures:
            creature.logic(self)
        for grass in self.grass:
            grass.update(self)
        # remove dead creatures
        creature: Creature
        for creature in self.creatures:
            if creature.is_dead():
                self.dead_creatures.append(creature)
                self.creatures.remove(creature)
        # remove dead grass
        grass: Grass
        for grass in self.grass:
            if grass.is_dead():
                self.grass.remove(grass)
        # cull the grass
        if len(self.grass) > Game._MAX_GRASS:
            self.grass = self.grass[:Game._MAX_GRASS]
        if len(self.creatures) == 1:
            self.creatures[0].kill()
            self.dead_creatures.append(self.creatures[0])
            self.restart()

        events = pygame.event.get()
        for event in events:
            # Detect a mouse click and select a debug creature
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = pygame.mouse.get_pos()
                debug_creature = None
                for creature in self.creatures:
                    if creature.is_clicked(x, y):
                        print("Clicked on creature")
                        self.debug_creature = creature
                        break
            if event.type == pygame.QUIT:
                self.running = False

    def init_game(self):
        Creature.set_game(self)
        self.init_grass()
        self.init_creatures()

    def init_creatures(self):
        for _ in range(Game._CREATURE_SPAWN_AMOUNT):
            x = random.randint(0, self.screen.get_width())
            y = random.randint(0, self.screen.get_height())
            new_cow = Creature(x, y)
            new_cow.mass = 50
            new_cow.set_photosynthesis(False)
            new_cow.set_sexual_reproduction(False)
            new_cow.set_speed(0.2)
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
        self.display_hud()
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

    def display_hud(self):
        x = 0
        y = 0
        font = pygame.font.SysFont("monospace", 15)
        debug_text = []
        if self.debug_creature is not None:
            debug_text = self.debug_creature_text()
        debug_text.extend(self.debug_common())
        self.print_text(x, y, font, debug_text)

    def print_text(self, x, y, font, debug_text) -> None:
        if debug_text is None:
            return
        for line in debug_text:
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (x, y))
            y += 15

    def debug_common(self) -> list:
        debug_text = []
        debug_text.append("Generation: " + str(self.generation))
        debug_text.append("Grass count: " + str(len(self.grass)))
        debug_text.append("Creature count: " + str(len(self.creatures))) 
        debug_text.append("Dead creature count: " + str(len(self.dead_creatures)))
        return debug_text

    def debug_creature_text(self) -> list:
        debug_text = []
        # Creature current actions:
        debug_text.append("Creature current action: " +
                          self.debug_creature._actions[self.debug_creature.action]["name"])
        # Creature coordinates:
        debug_text.append("Creature coordinates: " +
                          "X: " + str(self.debug_creature.x) + " Y: " + str(self.debug_creature.y))
        # Creature mass:
        debug_text.append("Creature mass: " + str(self.debug_creature.mass))
        return debug_text

    def restart(self):
        self.generation += 1
        self.creatures = []
        self.grass = []
        self.mutate_creatures()
        self.init_grass()
        
    def mutate_creatures(self):
        # sort dead creatures by age
        self.dead_creatures.sort(key=lambda x: x.age, reverse=True)
        # procreate every other creature with the next, twice
        for i in range(0, len(self.dead_creatures), 2):
            if i + 1 < len(self.dead_creatures):
                creature1 = self.dead_creatures[i]
                creature2 = self.dead_creatures[i + 1]
                creature1.procreate(creature2, self)
                creature2.procreate(creature1, self)
        self.dead_creatures = []

    def init_grass(self):
        for _ in range(Game._GRASS_SPAWN_AMOUNT):
            x = random.randint(0, self.screen.get_width())
            y = random.randint(0, self.screen.get_height())
            new_grass = Grass(x, y)
            self.add_grass(new_grass)