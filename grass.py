import random
import pygame

from drawable import Drawable


class Grass(Drawable):
    _photosynthesis_rate = 0.01

    def __init__(self, x: int, y: int, image: str = "images/grass.png"):
        super().__init__(x, y, image)
        self.age = 0
        self.max_age = 1000
        self.max_mass = 10
        self.dead = False
        self.redpoduction_age = 500
        self.spawn_count = 1
        self.spawn_radius = 10

    def draw(self, game_state):
        self.image = pygame.transform.scale(self.image, (self.mass, self.mass))
        game_state.screen.blit(self.image, (self.x, self.y))

    def update(self, game_state):
        self.age += 1
        if self.age > self.max_age:
            self.kill()
        if self.mass < self.max_mass:
            self.mass += Grass._photosynthesis_rate
        if self.age % self.redpoduction_age == 0:
            self.reproduce(game_state)
            
    def reproduce(self, game_state):
        # spawn spawn_count new grass in spawn_radius radius
        for _ in range(self.spawn_count):
            x = self.x + random.randint(-self.spawn_radius, self.spawn_radius)
            y = self.y + random.randint(-self.spawn_radius, self.spawn_radius)
            x = min(max(0, x), game_state.screen.get_width())
            y = min(max(0, y), game_state.screen.get_height())
            new_grass = Grass(x, y)
            game_state.add_grass(new_grass)
        

    def kill(self):
        self.dead = True

    def is_dead(self):
        return self.dead
