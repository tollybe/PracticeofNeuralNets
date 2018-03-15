"""
number = "12.34"
number1= 56.78
number2 = "2"
print (float(number), int(number1), int(number2))

F = 104

C = float(5)/9 * (F-32)
cel = 5/9 * float(F-32)
ce = float(5/9 * (F-32))

print (C, cel, ce, " are all examples of working equations")

## chapter 5
temp_input = float(input("Enter degrees in Fahrenheit:")) ## user input and converts unit into float
temp_output = float(5)/9 * (temp_input - 32) ## processes the input to effect
print (temp_input, " is ", temp_output, " in celcius.") ## prints out effect


##Chap 5 changing w + h to input

w = float(input(" enter the width of your room"))
h = float(input(" enter the length of your room"))
total = float(input(" How much per sqf?"))

area = w * h
yard = area *9
cost = total * area
perimeter = (2*w) + (2*h)
print("a = ", area, "of carpet (sqf)",yard, "- in yards"," needed",)
print("P - ", perimeter,"sqf", "it will cost you","£", cost, " in total")

##Chap 5 Of the above


distance = 200.0
speed = 80.0

time = distance/ speed

print (" it would take", time, " hours")

## Changed - see below print ("enter your name")
firstInput = input("enter your name")
lastName = input("and your surname?")
print ("hello,", firstInput, lastName, " how are you today?")

quarter = float(0.25)
dime = float(0.10)
nickel = float(0.05)
pennies = float(0.01)

quarterInput = float(input("how many quarters do you have?"))
dimeInput = float(input(" how many dimes do you have?"))
nickelInput = float(input(" How many nickels do you have?"))
penniesInput =  float(input("How many pennies do you have?"))

def coinCount():
    total = (quarterInput*quarter) + (dimeInput*dime) + (nickelInput*nickel) +(penniesInput+pennies)
    return total

print ("You have:", quarterInput, "quarter(s)", dimeInput, "dime(s)", nickelInput,"nickle(s)", "and",penniesInput ,"pennies",)
print ("in total, you have",coinCount(), "dollars worth of coins")

inputQ = input("enter a Q or q")
if inputQ =="Q" or " Q" or "Q ":
    print("It is capital")
elif inputQ == "q" or " q" or "q ":
    print("it is lowercase")
else:
    print("the scope only addresses capital and lowercase Q's")

priceCheck = float(input("What is the cost of the item?"))
if priceCheck > 0 and priceCheck <= 10 :
    print("purchase is 10% off")
    newPrice=(priceCheck * 1.1) + priceCheck
    print ("new cost is ", newPrice)
elif priceCheck > 11 and priceCheck <= 20 :
    print(" purchase is 20% off")
    newPrice =(priceCheck *1.2)+ priceCheck
else:
    print("no discount given")


genderPlayer =input("What is your gender?")
if genderPlayer =="f" or genderPlayer == "F" or genderPlayer =="female":
    agePlayer = int(input("and how old are you?"))
    if agePlayer >= 10 and agePlayer <= 12:
        print("You are eligible to join the team")
    elif agePlayer < 10:
        yearsLeft = 10 -agePlayer
        print ("try again in", yearsLeft, " years.")
    else:
        print("sorry you are too old.")
else:
    print("You are not eligible to join the team.")

import easygui

tankSize = float(input("How big is your petrol tank (in Liters)"))
bufferTankSize = tankSize - 5
tankVolume = float(input("How full is your tank (in percentage where 50% = 50)"))
percentageVolume = (bufferTankSize * tankVolume) / 100
tankDistance = float(input("How many KM per liter does your car get?"))
tankTravelTime = percentageVolume * tankDistance
tankPhrase = "You can go another " + str(tankTravelTime) + "km"
line = "The next petrol station is 200km away."

if tankTravelTime >= 200:
    easygui.msgbox(tankPhrase + "\n"
                   + line + "\n"
                   + " You can wait for the next station")
else:
    easygui.msgbox(tankPhrase + "\n"
                   + line + "\n"
                   + "Warning: Get petrol ASAP")


drowssap = str(1234)
randomWord = "Encrypted"
password = randomWord + drowssap
tries = 4


while tries != 0:
    guess = input("Enter Password:")
    if guess == password:
        print( "You're in!")
        tries = 0
    else:
        print("Incorrect.")
        tries = tries - 1
        print("You have " , tries , "tries remaining before you are locked out.")
        if tries == 0:
            print("You have been locked out.")


string = "Supercalifragelisticexpialidocious"
count = 0
for letter in string:
    count = count + 1
    print(letter)
print ("there are ",count," letters in",string)


import time

inputCountdown = int(input("How many seconds?"))

for seconds in range (inputCountdown,0, -1):
    for star in range(seconds):
        print("#")
    print(seconds, seconds * "*")
    time.sleep(1)
print ("Blast off!")

multiplyingNumber = int(input("Which multiplication table would you like?"))
multiplyingNumberLowRange = multiplyingNumber -3
multiplyingNumberHighRange = multiplyingNumber + 1

highMultiplier = int(input("How high do you want to go?"))
## for multiplier in range(multiplyingNumber-3, multiplyingNumber+1):
for multiplier in range(multiplyingNumberLowRange, multiplyingNumberHighRange):
    for i in range(0,highMultiplier +1):
        print(str(multiplier),"*", str(i),"=", i*multiplier) ##
    print()

i = 0
while i < 11:
    print(print(str(multiplyingNumber), "*", str(i), "=", i * multiplyingNumber))
    i = i + 1
    if i ==11:
        break

print ("\tDog \tBun \tKetchup \tMustard \tOnions \tCalories")
count = 1

dog_cal = 140
bun_cal = 120
mus_cal = 20
ket_cal = 80
onion_cal = 40

for dog in [0,1]:
    for bun in [0,1]:
        for ketchup in [0,1]:
            for mustard in [0,1]:
                for onion in [0,1]:
                    total_cal = (dog * dog_cal) + (bun * bun_cal) + (mustard * mus_cal) \
                                + (ketchup * ket_cal) + (onion * onion_cal)
                    print("#", count, "\t", dog, "\t", bun, "\t", ketchup, "\t", mustard, "\t", onion, "\t", total_cal, "\t")

                    count = count + 1
### 3 stars in 5 sets
for i in range(5):
    for j in range(3):
        print("*"),
    print()


numberOfDecisions = 0
for a in [0,1]:
    for b in [0,1]:
        for c in [0,1]:
            for d in [0,1]:
                numberOfDecisions= numberOfDecisions + 1
print(numberOfDecisions)
print( 2**4 )
##2**4 = 16

nameList = []
for numberOfName in range(1,6):
    inputName = input("insert a name:")
    nameList.append(inputName)
print("The names are ", ", ".join(nameList)),

print ("The third name you entered is: ",nameList[2]),

sortNameList = sorted(nameList)
##nameList.sort()
print("The names sorted are ", ", ".join(sortNameList))

selectFromList =int(input(" Replace a name. Which one? (1-5)"))
replaceName = input("New Name:")
nameList[selectFromList-1] = replaceName
print("The names list is now ", ", ".join(nameList))

dictionary = {}
unavailable = "That word isn't in the dictionary yet"
addOrLook = ()

while addOrLook != 3:
    addOrLook = int(input("Add(enter '1') or look up a word(enter '2')? To end, press 3: "))
    if addOrLook == 1:
        wordInput = input("Enter the word: ")
        definitionInput = input("Enter the definition:"),
        dictionary[wordInput] = definitionInput
        print("Word added!")
    elif addOrLook == 2:
        searchInput = input("Search word:")
        if searchInput in dictionary.keys():
            print(dictionary[searchInput])
        else:
            print(unavailable)
    elif addOrLook == 3:
        break
    else:
        print("invalid character entry. Type '1' or '2' or '3'.")



myname = []

def myaddress(myname):
    print (myname)
    print ("This is an address")


print (myaddress("Tolly"))

def calculateTax(price, taxRate):
    total = price + (price * taxRate)
    return total

global myPrice = float(input("Enter a price: "))

totalPrice = calculateTax(myPrice, 0.179)
print ("Price = ", myPrice, "Total price = ", totalPrice)

letters ={
    (T:


    )


}

def anyAddress(name,address,city,province,postcode):
    print(name),
    print(address),
    print (city),
    print(province),
    print(postcode)
print ("Comatose", " Somewhere", "nowhere City", "nowhereLand" "N0WH3R3")


class BankAccount:

    def __init__(self,accountName,accountNumber,balance):
        self.name = accountName
        self.accountNumber = accountNumber
        self.balance = balance

    def __str__(self):
        msg = "Bank account" + ",name = ", + self.name + ",account number" + self.accountNumber
        return msg

    def displayBalance(self):
        print("Current balance:", self.balance)

    def makeDeposit(self, deposit):
        self.balance = self.balance + deposit
        print("You have deposited:", deposit, ". Current balance:", self.balance)

    def withdrawal(self,withdraw):
        self.balance = self.balance - withdraw
        print("You have drawn:", withdraw,". Current balance:", self.balance)


##come back to this tomorrow
class InterestAccount(BankAccount):

    def __init__(self,interest, accountName,accountNumber,balance):
        BankAccount.__init__(self,accountName,accountNumber,balance)
        self.interest = interest

    def addInterest(self):
        rate = 1 + (self.interest/100.0) ## do not use field from constructor as assign
        frenchToast = self.balance * rate ## use self. blah to modify object
        self.balance = frenchToast
        print("Interest yields: ",self.balance, "added to account")


account = BankAccount("Tom","1234Abc",245)
print (account)
account.displayBalance()
account.makeDeposit(400)
account.withdrawal(20)

##irrelevant to BankAccount object
interestAcc = InterestAccount(7,"Sarah", "1234Abc", 245)
interestAcc.withdrawal(200)

interestAcc.addInterest()


import python_module
celcius = float(input(" Enter a temperature in Celcius: "))
fahrenheit = python_module.c_to_f(celcius)
print("That's", fahrenheit, " degrees in F")

from python_module import c_to_f

celcius = float(input("Enter degrees in clecius: "))
fahrenheit = c_to_f(celcius)
print( "yeah bitches", fahrenheit, "f degrees")


import random
from time import sleep
from statistics import mode

newlist = []
for i in range(0,5):
    newNumber = random.randint(1,20)
    newlist.append(newNumber)
    print(newlist)

lengthList = []
for i in range (0,3):
    sleepSpecification = random.randint(3, 10)
    print(" please wait", sleepSpecification, " seconds for a decimal.")
    sleep(sleepSpecification)
    decimal = random.random()
    print("decimal selected: ",decimal)
    lengthList.append(sleepSpecification)
total = sum(lengthList)
average = float(total/len(lengthList))
print ("A total of",total," seconds of your life wasted. Averaging:", average, " seconds.")


import pygame,sys,random
from pygame.color import THECOLORS
pygame.init()
screen = pygame.display.set_mode([640,480])
###Step 1 ^ in word order
running = True
### Step 2.0^ continuously runs
screen.fill([255,255,255])
for i in range(10):
    width = random.randint(0,125); height = random.randint(0,216)
    top = random.randint(0,343); left = random.randint (0,343)
    colourName = random.choice(list(THECOLORS.keys()))
    colour = THECOLORS[colourName]
    lineWidth = random.randint(1,8)
    #red = random.randint(0,255)
    #green = red
    #blue = red
    #colour = [red,green,blue]
    rectParams = [width,height,top,left]
    pygame.draw.rect(screen, colour, rectParams, lineWidth)
pygame.display.flip()
## Step 3 ^ make white screen + first image (on screen draw colour, location, size + filled

### Step 2.1 v allowing x box to be closed
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
pygame.quit()


import pygame, sys
import math

pygame.init()
screen = pygame.display.set_mode([640,480])

screen.fill ([255,255,255])
sin1plotPoints =[]
sin2plotPoints =[]
for x in range(0,640):
    y = int(math.sin(x/640 * 4 * math.pi)* 200 + 240)
    z = int(math.sin(x/640 * 2**3 * math.pi) *200 + 240)
    sin1plotPoints.append([x, y])
    sin2plotPoints.append([x, z])
    ## draws wave - pygame.draw.rect(screen, [255,0,0], [x,y,1,1], 1)
pygame.draw.lines(screen, [0,255,0], False, sin1plotPoints, 5) #draws dot to dot wave
pygame.draw.lines(screen, [255,0,0], False, sin2plotPoints,3)
pygame.display.flip()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.time.delay(30)

pygame.quit()

#chapter 16 + 18 combined
import pygame, sys

pygame.init()
screen = pygame.display.set_mode([640,480])
background = pygame.Surface(screen.get_size())
background.fill([255,255,255])
clock = pygame.time.Clock()


class Ball (pygame.sprite.Sprite):
    def __init__(self,image_file, speed, location):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location
        self.speed = speed

    def move(self):
        if self.rect.left <= screen.get_rect().left or self.rect.right >= screen.get_rect().right:
            self.speed[0] =- self.speed[0]
        newPos = self.rect.move(self.speed)
        self.rect = newPos


myWrapBall = Ball("my_ball.png", [10,0],[20,20])
delay = 100
interval = 20
pygame.key.set_repeat(delay,interval)
held_down = False
pygame.time.set_timer(pygame.USEREVENT, 10)
direction = 1

#myWrapBall = pygame.image.load("my_ball.png")
#x = 50
#y = 50
#screen.blit(myWrapBall,[x,y])
#pygame.display.flip()

#myBounceBall = pygame.image.load("my_ball.png")
#xOne = 50
#z = 350
#xSpeed = 10
#zSpeed = 5


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        ## Press mouse where mouse has priority
        elif event.type == pygame.MOUSEBUTTONDOWN:
            held_down = True
        elif event.type == pygame.MOUSEBUTTONUP:
            held_down = False
        elif event.type == pygame.MOUSEMOTION:
            if held_down:
                myWrapBall.rect.center = event.pos
         ## Press button
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                myWrapBall.rect.top = myWrapBall.rect.top - 10
            elif event.key == pygame.K_DOWN:
                myWrapBall.rect.top = myWrapBall.rect.top + 10
        elif event.type ==pygame.USEREVENT:
            myWrapBall.rect.centery = myWrapBall.rect.centery + (30*direction)
            if myWrapBall.rect.top <=0 or myWrapBall.rect.bottom>= screen.get_rect().bottom:
                direction =- direction


    clock.tick(30)
    screen.blit(background,(0,0))
    myWrapBall.move()
    screen.blit(myWrapBall.image, myWrapBall.rect)
    pygame.display.flip()

    #pygame.time.delay(30)
    #pygame.draw.rect(screen, [255, 255, 255], [xOne, z, 100, 100], 0) ##bounceball
    #pygame.draw.rect(screen, [255, 255, 255], [x, y, 100, 100], 0) ##wrapp ball
    #x = x + xSpeed
    #xOne = xOne + xSpeed
    #z = z + zSpeed
    #if xOne > (screen.get_width() - 100)/2 or xOne < 0 + 50:
    #    xSpeed =- xSpeed
    #if x > screen.get_width() + 100:
    #    x = -100
    #if z > (screen.get_height() - 100)/2 or z <0 +54:
    #    zSpeed =- zSpeed
    #screen.blit(myBounceBall,[xOne,z])
    #screen.blit(myWrapBall, [x,y])
    #pygame.display.flip()

pygame.quit()

##17
import pygame, sys
from random import *


class MyBallClass (pygame.sprite.Sprite):
    def __init__(self, image_file, location, speed):
        pygame.sprite.Sprite.__init__(self) #initialises sprite
        self.image = pygame.image.load(image_file) #loads image file
        self.rect = self.image.get_rect() #defines boundaries through rect
        self.rect.left, self.rect.top = location #sets the initial location of the ball
        self.speed = speed

    def move(self):
        self.rect = self.rect.move(self.speed)
        if self.rect.left < 0 or self.rect.right > width:
            self.speed[0] =- self.speed[0]
        if self.rect.top < 0 or self.rect.bottom > height:
            self.speed[1] =- self.speed[1]


def react(group):
    screen.fill([255,255,255])
    for ball in group:
        ball.move()
    for ball in group:
        group.remove(ball)

        if pygame.sprite.spritecollide(ball,group,False):
            ball.speed[0] =- ball.speed[0]
            ball.speed[1] =- ball.speed[1]

        group.add(ball)
        screen.blit(ball.image,ball.rect)
    pygame.display.flip()
    #pygame.time.delay(20)

size = width, height = 640, 480
screen = pygame.display.set_mode(size)
screen.fill([255,255,255])
img_file = "my_ball.png"
clock = pygame.time.Clock()
group = pygame.sprite.Group()
balls = []
for row in range (0,2):
    for column in range (0,2):
        location = [column * 180 + 10, row * 180 + 10]
        speed = [choice([-2.5,2.5]), choice ([-2.5,2.5])]
        ball = MyBallClass(img_file,location, speed)
        group.add(ball) #adds ball to the group
        #balls.append(ball)


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            frame_rate = clock.get_fps()
            print("Frame rate =", frame_rate)
    react(group)
    clock.tick(60)  # ( times per second the loop should run - fps)



    #pygame.time.delay(20)
    #screen.fill([255,255,255])
    #for ball in balls:
    #    ball.move()
    #    screen.blit(ball.image, ball.rect)
    #pygame.display.flip()

pygame.quit()

##with this program, make a path finding for each individual sprite and have them learn the fast route to get to it.


import pygame

pygame.init()
pygame.mixer.init()

screen = pygame.display.set_mode([640,480])
pygame.time.delay(1000)

#wavs
splat = pygame.mixer.Sound("splat.wav")
splat.set_volume(.5)
splat.play()

#music mp3s etc.
pygame.mixer.music.load("bg_music.mp3")
pygame.mixer.music.set_volume(.3)
pygame.mixer.music.play()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not pygame.mixer.music.get_busy():
        splat.play()
        pygame.time.delay(1000)
        running = False
pygame.quit()


#chapter 20 - COME BACK TO

import sys
from PyQt5 import QtCore, QtGui,QtWidgets, uic

form_class = uic.loadUiType("MyFirstGui.ui")[0]


class MyWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self,parent=None):
        QtWidgets.QMainWindow.__init__(self,parent)
        self.setupUi(self)

app = QtGui.QGuiApplication(sys.argv)
myWindow = MyWindowClass()
myWindow.show()
app.exec_()


print ("ABC\tXYZ")

print ("Number \tSquare \tCube")
for i in range (1,11):
    print(i, "\t","\t", i**2,"\t", "\t", i**3)

name1= "tollybe"

name = "TB,homer,Jesus,gandhi, hitler, cheese"
parts = name.split("hitler")
for part in parts:
    print (part + " ")

longlist = ["why", "am", "I", "bored", "?"]
longString = " ".join(longlist)
print(longString)

short_name = name1.strip("be")
print (short_name)

caps = name1.upper()
print(caps)
lower = caps.lower()
print(caps + lower)


age = 13
teen = 14
print(" I have a %s and I worked dis out before" %short_name) # %s for string
print( " I am not attracted to %i year olds." %age) # %i or %d for int
print( " average was %.2f" %age) # %f for float where .x = degrees of freedom
print ("%+.1f" %age)
print ("%e," %age) #%e or %g (autmoatic notation) for exponents ^ 10
print( "%i%%" %age) # repeat after signals e.g. \\ for "\" or %% for "%"
print ("I got {0:1f}% in math and {1:.1f}% in science".format(age,teen))



number = int(input("Print number you want fraction"))
fraction = 1.0/number
print ("Number","\t","Fraction")

for i in range(0,number + 1):
    newFraction = fraction * i
    print("#", i, "\t", "%.3f"%newFraction)


import pickle

my_photo = open("C:/Users/Tolly/Dropbox/University/Unrelated to Psychology/Photo/Tolly.jpg","r")

#"r" = reading, "w" = writing (it replaces or creates new file), "a" = appending, "rb" for reading binary files
notes = open("notes.txt","r")
firstLine = notes.readline()
secondLine = notes.readline()
thirdLine = notes.readline()
forthLine = notes.readline()
fifthLine = notes.readline()
#lines = notes.readlines()

print (firstLine,secondLine,thirdLine,forthLine,fifthLine)
newNote = open("notes.txt","a")
newNote.write("\nWorkout my plan to exist")
newNote.close()

print(notes)

notes.seek(50)
print(notes.readline())
notes.close()

music = open("splat.wav","rb")

test = ["frank", 73,"boop",91.87e18]
pickleFile = open("my_pickled_list.pkl","w")
pickle.dump(test,pickleFile)
pickleFile.close()

import random

adjectives = open("adjectives.txt","r")
#adjectivesLine = adjectives.readline()
adjectiveWord = adjectives.split(", ")
adjectives.close()
randomAdjective = random.choice(adjectiveWord)

nouns = open("nouns.txt","w")
nouns.write("monkey\nelephant\ncyclist\nteacher\nauthor\hockey player")
#nounsLine = nouns.readline()
nounWord = nouns.split(", ")
nouns.close()
randomNoun = random.choice(nounWord)

verb = open("verbs.txt","w")
verb.write("played a ukelele\ndanced a jig\ncombed his hair\nflapped her ears")
#verbLine = verb.readline()
verbPhrase = verb.split(", ")
randomVerb = random.choice(verbPhrase)

adverb = open("adverb.txt", "w")
adverb.write("on a table\nat the grocery store\nin the shower\nafter breakfast\n with a broom")
#adverbLine = adverb.readline()
adverbPhrase = adverb.split()
randomAdverb = random.choice(adverbPhrase)

print ("The ", randomAdjective,randomNoun,randomVerb,randomAdverb,".")
"""
#saving + storng
import pickle
name= input("Enter your name: ")
age = input("enter you age: ")
colour = input("enter your favourite colour: ")
food = input("enter your favourite food: ")

list = [name,age,colour,food]
pickle_file = open("my_pickle_file.pkl","w")
pickle.dump(list, pickle_file)
pickle_file.close

#data = open ("data.txt","w")
#data.write(name + "\n" + str(age) + "\n" + colour + "\n" + food)
#data.close()
