import pygame, sys
import random
image= pygame.image.load("Background_Sheep.png")

#class that defines the object pingpongball
class PingPongBallClass(pygame.sprite.Sprite):
    def __init__(self, image_file, speed, location):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location
        self.speed = speed

    def move(self):
        global score, scoreFont,scoreSurf
        self.rect = self.rect.move(self.speed)

        if self.rect.left < 0 or self.rect.right > screen.get_width():
            self.speed[0] =- self.speed[0]
            global hitWallList
            hitWallList= pygame.mixer.Sound(random.choice(["baa_1.wav", "baa_2.wav"]))
            hitWallList.set_volume(0.4)
            hitWallList.play()
        if self.rect.top <=0:
            self.speed[1] =- self.speed[1]
            score = score + 1
            scoreSurf = scoreFont.render(str(score),1,(255,0,0))
            if score > 10:
                self.speed[1] = self.speed[1] * speedMultiplierX ## needs to change

        # increase game difficulty depending on score by increasing ball speed

        """
        ## things to improve upon
        difficulty = speed change [1] 
        
        if score increments in 10:
        
        difficulty increases on the condition that it is same life 
        
        and so on and so forth
        
        if dies and still have life left, difficulty is lowered, speed is dropped to previous difficulty setting
        
        note: difficulty appears next to scoreSurf value 
        
        """

#class that defines the object paddle
class PaddleClass(pygame.sprite.Sprite):
    def __init__(self,location):
        pygame.sprite.Sprite.__init__(self)
        image_surface = pygame.surface.Surface([100,20])
        image_surface.fill([0,0,0])
        self.image = image_surface.convert()
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location


#setting up main file
pygame.init()
screen = pygame.display.set_mode([640,480])
clock = pygame.time.Clock()
ball_speed = [random.randint(4,7),random.randint(3,5)]
speedMultiplierX = 1.05
score = 0
lives = 3


##creating object class
pingpongball = PingPongBallClass("sheep_ball.png", ball_speed,[50,50])
ballGroup = pygame.sprite.Group(pingpongball)
paddle = PaddleClass([270,400])

##creating font
scoreFont = pygame.font.Font(None,50)
scoreSurf = scoreFont.render(str(score),1,(255,0,0))
scorePosition = [10,10]

#running the file
done = False
running = True

#MUSIC
#BEEP BEEP I'M A SHEEP!  Change colours every 10 seconds
pygame.mixer.music.load("beep_beep_sheep.mp3")
pygame.mixer.music.set_volume(.09)
pygame.mixer.music.play(-1)

#sounds
hitPaddle = pygame.mixer.Sound("quack.wav")
hitPaddle.set_volume (0.4)
nuu = pygame.mixer.Sound("Nuu.wav")
nuu.set_volume(0.7)
newLife = pygame.mixer.Sound("new_life.wav")
newLife.set_volume(0.3)
gameOver = pygame.mixer.Sound("fuck.wav")
gameOver.set_volume(0.3)

pygame.time.set_timer(pygame.USEREVENT,10000)
colour = [random.randint(0,255),random.randint(0,255), random.randint(0,255)]

while running:
    clock.tick(60)
    screen.fill(colour)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            frame_rate = clock.get_fps()
            print("Frame rate =", frame_rate)
        elif event.type == pygame.MOUSEMOTION:
            paddle.rect.centerx = event.pos[0]
        if event.type == pygame.USEREVENT:
            colour = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

    if not done:
        # collision between pong + paddle
        if pygame.sprite.spritecollide(paddle, ballGroup, False):
            pingpongball.speed[1] = - pingpongball.speed[1]
            hitPaddle.play()

        pingpongball.move()

        screen.blit(image,(0,0))
        screen.blit(pingpongball.image, pingpongball.rect)
        screen.blit(paddle.image, paddle.rect)
        screen.blit(scoreSurf, scorePosition)

        ##Lives left visible to user
        for i in range(lives):
            width = screen.get_rect().width
            screen.blit(pingpongball.image, [width - 40 * i, 20])
        pygame.display.flip()


    #if ball falls off the edge : a. GameOver b. Continue
    if pingpongball.rect.top >= screen.get_rect().bottom:
        lives = lives - 1
        nuu.play()
        if lives == 0:
            nuu.stop()
            newLife.stop()
            hitWallList.stop()
            hitPaddle.stop()
            pygame.mixer.music.stop()
            pygame.time.delay(1000)
            gameOver.play()
            final_text1 = "Game Over!"
            final_text2 = "Your final score is: " + str(score)
            final_text3 = "That's a terrible score. You can do better than that."
            ft1Font = pygame.font.Font(None, 100)
            ft1Surf = ft1Font.render(final_text1, 1, (255, 0, 0))
            ft2Font = pygame.font.Font(None, 50)
            ft2Surf = ft2Font.render(final_text2, 1, (0, 255, 0))
            ft3Font = pygame.font.Font(None, 20)
            ft3Surf = ft3Font.render(final_text3, 1, (0, 255, 0))

            screen.blit(ft1Surf, [screen.get_width() / 2 - ft1Surf.get_width() / 2, 100])
            screen.blit(ft2Surf, [screen.get_width() / 2 - ft2Surf.get_width() / 2, 200])
            screen.blit(ft3Surf, [screen.get_width() / 2 - ft3Surf.get_width() / 2, 450])
            pygame.display.flip()
            done = True
        else:
            newLife.play()
            pygame.time.delay(1500)
            pingpongball.rect.topleft = [50,50]

pygame.quit()
