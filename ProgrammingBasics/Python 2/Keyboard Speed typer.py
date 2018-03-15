import datetime
"""
past = datetime.datetime(year = 2018, month = 01,day = 25,hour = 18,minute =07, second= 00)
print past.ctime()

today = datetime.date(year = 2019,month = 02, day = 27)
someTime = datetime.time(hour = 18, minute = 27, second = 43)


print today, someTime

present = datetime.datetime.combine(today,someTime)

print present

difference = present - past
print type(difference), difference

print datetime.datetime.now()


#creating a WPM game
import time, random
# List of messages that will spawn
messages = [
    "Of all the trees we could've hit, we had to get one that hits back.",
    "If he doesn't stop trying to save your life he's going to kill you.",
    "It is our choices that show what we truly are, far more than our abilities.",
    "I am a wizard, not a baboon brandishing a stick.",
    "Greatness inspires envy, envy engenders spite, spite spawns lies.",
    "In dreams, we enter a world that's entirely our own.",
    "It is my belief that the truth is generally preferable to lies.",
    "Dawn seemed to follow midnight with indecent haste."
    ]
#Preempting user
print "Typing speed test. Type the following message. I will time you"
time.sleep(2)
print "\nReady.."
time.sleep(1)
print "\nGO!"
time.sleep(1)
#selecting random message from messages
message = random.choice(messages)
print  "\n" + message
startTime = datetime.datetime.now()
#user input
typing = raw_input(">")
endTime = datetime.datetime.now()
difference = endTime - startTime
typingTime = difference.seconds + difference.microseconds/float(1000000)
#calculations
cps = len(message)/typingTime
wpm = cps * 60/5.0
print"\nYou typed %i characters in %.1f seconds"%(len(message),typingTime)
print " That's %.2f chars per sec, or %.1f words per minute" %(cps,wpm)
if typing == message:
    print " no mistakes"
else:
    print" But mistakes were made."

"""

import pickle, os

firstTime = True
if os.path.isfile("last_run.pkl"):
    pickle_file = open("last_run.pkl", 'r')
    last_time = pickle.load(pickle_file)
    pickle_file.close()
    print " The last time this program was run was", last_time
    firstTime = False

pickle_file = open("last_run.pkl", 'w')
pickle.dump(datetime.datetime.now(), pickle_file)
pickle_file.close()
if firstTime:
    print("Created new pickle file.")


try:
    file = open("somefile.txt", 'r')
except:
    print ("couldn't open the file. Do you want to reenter the filename?")