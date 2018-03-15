import pygame

# Making throttle class
class ThrottleClass(pygame.sprite.Sprite):
    def __init__(self, location = [0,0]):
        pygame.sprite.Sprite.__init__(self)
        image_surface = pygame.surface.Surface([30,10])
        image_surface.fill([128,128,128])
        self.image = image_surface.convert()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.centery = location

#calculate position, motion, acceleration, fuel
def calculate_velocity():
    global thrust,fuel,velocity, delta_v,height,y_pos