import random
import sys
import pygame

from creature import Creature


class Game:
    def __init__(self):
        self._screenshot_filename = "screenshot.jpg"
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
            self.draw()
            self.logic()
            pygame.display.flip()

    def logic(self):
        # make a screenshot
        pygame.image.save(self.screen, self._screenshot_filename)
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
            new_grass.set_photosynthesis(True)
            new_grass.set_sexual_reproduction(False)
            self.creatures.append(new_grass)
        x = random.randint(0, self.screen.get_width())
        y = random.randint(0, self.screen.get_height())
        new_cow = Creature(x, y, self.screen)
        new_cow.mass = 50
        new_cow.set_photosynthesis(False)
        new_cow.set_sexual_reproduction(True)
        self.creatures.append(new_cow)

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
