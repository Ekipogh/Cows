import pytest

from cows import Creature


class TestCreature:
    creature = Creature(0, 0, None)

    def test_zero_speed(self):
        assert self.creature.get_speed() == 0

    def test_nonzero_speed1(self):
        self.creature.genome[0] = 1
        assert self.creature.get_speed() == 1

    def test_zero_photo(self):
        assert self.creature.get_photosynthesis() == 0

    def test_nonzero_photo(self):
        self.creature.genome[1] = 1
        assert self.creature.get_photosynthesis() == 1
