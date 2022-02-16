import sys
import pygame


class Creature:
    def __init__(self, x, y, screen, image: str):
        if image is None:
            image = "images/error.png"
        self.screen = screen
        self.x = x
        self.y = y
        self.im = pygame.image.load(image)
        self.rect = self.im.get_rect()
        self.rect.move(self.x, self.y)
        self.genome = []

    def get_speed(self):
        try:
            return self.genome[0]
        except IndexError:
            return 0

    def draw(self):
        self.screen.blit(self.im, self.rect.move([self.x, self.y]))

    def logic(self):
        pass


class Grass(Creature):
    def __init__(self, x, y, screen):
        super().__init__(x, y, screen, image="images/grass.png")


class Cow(Creature):
    def __init__(self, x, y, screen):
        super().__init__(x, y, screen, image="images/cow.png")

    def logic(self):
        pass


class Game:
    def __init__(self):
        pygame.init()
        self._size = 1024, 768
        self._black = 0, 0, 0
        self.screen = pygame.display.set_mode(self._size)
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
        self.creatures = []
        grass = Grass(0, 0, self.screen)
        cow = Cow(0, 64, self.screen)
        self.creatures.append(grass)
        self.creatures.append(cow)

    def draw_creatures(self):
        for creature in self.creatures:
            creature.draw()

    def draw(self):
        self.draw_creatures()


if __name__ == '__main__':
    game = Game()
    game.loop()
