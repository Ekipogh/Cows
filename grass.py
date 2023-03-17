import pygame

from drawable import Drawable


class Grass(Drawable):
    _photosynthesis_rate = 1

    def __init__(self, x: int, y: int, image: str = "images/grass.png"):
        super().__init__(x, y, image)
        self.dead = False

    def draw(self, game_state):
        self.image = pygame.transform.scale(self.image, (self.mass, self.mass))
        game_state.screen.blit(self.image, (self.x, self.y))

    def update(self):
        self.mass += Grass._photosynthesis_rate

    def kill(self):
        self.dead = True

    def is_dead(self):
        return self.dead
