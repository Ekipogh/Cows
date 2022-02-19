import random
import sys
import pygame


class Creature:
    def __init__(self, x, y, screen, image: str = "images/blank.png"):
        self.screen = screen
        self.x = x
        self.y = y
        self.im: pygame.Surface = pygame.image.load(image)
        self.rect = self.im.get_rect()
        self.rect.move(self.x, self.y)
        self.genome = [None] * 2

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
        image.fill((0, greenness, 0, 255), special_flags=pygame.BLEND_RGBA_ADD)
        self.screen.blit(image, (self.x, self.y))

    def logic(self):
        self.x = self.x + self.get_speed()


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
        for creature in self.creatures:
            creature.logic()

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
            self.creatures.append(new_grass)

    def draw_creatures(self):
        for creature in self.creatures:
            creature.draw()

    def draw(self):
        self.draw_creatures()


if __name__ == '__main__':
    game = Game()
    game.loop()
