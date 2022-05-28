import yappi

from game import Game

if __name__ == '__main__':
    yappi.start()
    game = Game()
    game.loop()
    yappi.get_func_stats().print_all()
    yappi.get_thread_stats().print_all()
