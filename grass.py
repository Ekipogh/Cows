import pygame


class Grass:
    _photosynthesis_rate = 1

    def __init__(self, x: int, y: int, image: str = "images/grass.png"):
        self.x = x
        self.y = y
        self.image: pygame.Surface = pygame.image.load(image)
        rect = self.image.get_rect()
        rect.move(self.x, self.y)
        self.mass = 0
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
