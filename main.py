from calendar import c
from logging import config
import netrc
from os import pipe
from turtle import st
import pygame
import sys
import random
import os
import neat

pygame.init()

WIDTH = 576
HEIGHT = 1024

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

TOP_PIPE = 0
BOTTOM_PIPE = 1
PIPE_GAP = 200

scroll_speed = 5

bird_image = pygame.transform.scale(pygame.image.load('sprites/yellowbird-downflap.png'), (68, 48))
ground_image = pygame.transform.scale(pygame.image.load('sprites/base.png'), (576, 224))
bottom_pipe_image = pygame.transform.scale(pygame.image.load('sprites/pipe-green.png'), (104, 640))
top_pipe_image = pygame.transform.flip(bottom_pipe_image, False, True)

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = bird_image
        self.rect = self.image.get_rect()
        self.rect.center = (100, 450)
        self.velocity = 0
        self.tilt = 0
        self.gravity = 0.7
        self.score = 0

    def update(self):
        self.velocity += self.gravity
        self.rect.y += int(self.velocity)    

        self.tilt = self.velocity * -7
        if self.tilt < -65:
            self.tilt = -65
        if self.rect.y >= HEIGHT - 224:
            self.rect.y = 400
            self.velocity = 0
        
        self.image = pygame.transform.rotate(bird_image, self.tilt)
        self.rect = self.image.get_rect(center=self.rect.center)
    
    def jump(self):
        self.velocity = -10
    
    def reset(self):
        self.rect.center = (100, 450)
        self.velocity = 0

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, direction):
        pygame.sprite.Sprite.__init__(self)

        if direction == TOP_PIPE:
            self.image = top_pipe_image
        else:
            self.image = bottom_pipe_image

        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y  

    def update(self):
        self.rect.x -= 5
        if self.rect.x <= -(self.image.get_width()+10):
            self.kill()
        
class Background:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.image = pygame.transform.scale(pygame.image.load('sprites/background-day.png'), (WIDTH, HEIGHT))

    def draw(self):
        screen.blit(self.image, (self.x, self.y))

class Ground(pygame.sprite.Sprite):
    def __init__(self, x):
        pygame.sprite.Sprite.__init__(self)
        self.image = ground_image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = HEIGHT - ground_image.get_height()

    def update(self):
        self.rect.x -= scroll_speed
        if self.rect.x <= -WIDTH:
            self.kill()


def main(genomes, config):
    running = True

    birds = []
    nets = []
    ge = []

    pipes = pygame.sprite.Group()
    ground = pygame.sprite.Group()
    bird_group = pygame.sprite.Group()

    for _,genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        bird = Bird()
        birds.append(bird)
        bird_group.add(bird)
        genome.fitness = 0
        ge.append(genome)

    ground.add(Ground(0),Ground(WIDTH))   

    background = Background()

    pipe_timer = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()

        ground.update()
        bird_group.update()

        if pipe_timer <= 0:
            new_y = random.randint(-500, -100)
            pipes.add(Pipe(WIDTH, new_y, TOP_PIPE), Pipe(WIDTH, new_y + top_pipe_image.get_height() + PIPE_GAP, BOTTOM_PIPE))
            pipe_timer = 100

        pipe_timer -= 1

        pipes.update()

        if len(ground) < 2:
            ground.add(Ground(WIDTH - scroll_speed))

        if len(birds) == 0:
            running = False
            break

        target_pipes = pipes.sprites()[:2] 
        if len(pipes) == 4:
            if pipes.sprites()[0].rect.x + top_pipe_image.get_width() < bird_group.sprites()[0].rect.x:
                target_pipes = pipes.sprites()[2:4]       

        for x,bird in enumerate(birds):
            collision_pipes = pygame.sprite.spritecollide(bird, pipes, False)
            collision_ground = pygame.sprite.spritecollide(bird, ground, False)
            if collision_pipes or collision_ground or bird.rect.y < 0:
                ge[x].fitness -= 1
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
                bird_group.remove(bird)
                continue

            ge[x].fitness += 0.05

            output = nets[x].activate((bird.rect.centery, abs(bird.rect.y - target_pipes[0].rect.y + top_pipe_image.get_height()), abs(bird.rect.y - target_pipes[1].rect.y)))
            if output[0] > 0.5:
                bird.jump()

            if len(pipes) > 0 and bird.rect.centerx > pipes.sprites()[0].rect.centerx and bird.rect.centerx < pipes.sprites()[0].rect.centerx + 5:
                bird.score += 1
                ge[x].fitness += 5
        

        background.draw()

        pipes.draw(screen)
        ground.draw(screen) 
        bird_group.draw(screen)

        pygame.display.flip()
        clock.tick(60)

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(main, 100)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)