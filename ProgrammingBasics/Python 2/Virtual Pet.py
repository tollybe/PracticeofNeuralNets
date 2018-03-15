import sys, datetime, pickle,easygui
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QInputDialog

formclass, baseclass = uic.loadUiType("mainwindow.ui")


class MyPet(baseclass, formclass):
    # constructing the object
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        # setting up buttons
        self.actionFeed.triggered.connect(self.feed_Click)
        self.actionWalk.triggered.connect(self.walk_Click)
        self.actionPlay.triggered.connect(self.play_Click)
        self.actionDoctor.triggered.connect(self.doctor_Click)
        self.actionStop.triggered.connect(self.stop_Click)
        self.actionRename.triggered.connect(self.rename_Click)
        self.actionPause.triggered.connect(self.pause_Click)

        # setting all variables false
        self.eating = False
        self.walking = False
        self.playing = False
        self.doctor = False
        self.sleeping = False
        self.forceAwake = False
        self.pause = False

        # building factors that affect MyPet class
        self.time_cycle = 0
        self.hunger = 0
        self.happiness = 8
        self.health = 8
        self.name = ""

        # images
        self.sleepImages = ["sleep1.gif", "sleep2.gif", "sleep3.gif", "sleep4.gif"]
        self.eatImages = ["eat1.gif", "eat2.gif"]
        self.walkImages = ["walk1.gif", "walk2.gif", "walk3.gif", "walk4.gif"]
        self.playImages = ["play1.gif", "play2.gif"]
        self.doctorImages = ["doc1.gif", "doc2.gif"]
        self.nothingImages = ["pet1.gif", "pet2.gif", "pet3.gif"]

        # default image
        self.imageList = self.nothingImages
        self.imageIndex = 0

        # setting up timer to cycle through animation
        self.myTimerAnim = QtCore.QTimer(self)
        self.myTimerAnim.start(500)
        self.myTimerAnim.timeout.connect(self.animation_timer)
        # setting up timer to cycle through clicks ( 60 clicks = 1 day)
        self.myTimerClick = QtCore.QTimer(self)
        self.myTimerClick.start(5000)
        self.myTimerClick.timeout.connect(self.tick_timer)

        # setting up a save file and program whilst closed and exception handling it
        filehandle = True
        try:
            file = open("savedata_vp.pkl", 'r')
        except:
            filehandle = False
        if filehandle:
            save_list = pickle.load(file)
            file.close()
        else:
            save_list = [8, 8, 0, datetime.datetime.now(), 0, "unnamed", False]
            self.happiness = save_list[0]
            self.health = save_list[1]
            self.hunger = save_list[2]
            timestamp_then = save_list[3]
            self.time_cycle = save_list[4]
            self.name = save_list[5]
            self.pause = save_list[6]
            if self.pause:
                self.actionPause.setText("Resume")
                QtGui.QIcon.Active

            else:
                self.actionPause.setText("Pause")
                QtGui.QIcon.Off


            # check if program is paused
            if not self.pause:
                # check how long since program was last run
                difference = datetime.datetime.now() - timestamp_then
                ticks = difference.seconds / 50
                for i in range(0, ticks):
                    self.time_cycle += 1
                    if self.time_cycle == 60:
                        self.time_cycle = 0
                    # awake
                    if self.time_cycle <= 48:
                        self.sleeping = False
                        if self.hunger < 8:
                            self.hunger += 1
                    # asleep
                    else:
                        self.sleeping = True
                        if self.hunger < 8 and self.time_cycle % 3 == 0:
                            self.hunger += 1
                    if self.hunger == 7 and (self.time_cycle % 2 == 0 and self.health > 0):
                        self.health -= 1
                    if self.hunger == 9 and self.hunger > 0:
                        self.health -= 1
                if self.sleeping:
                    self.imageList = self.sleepImages
                else:
                    self.imageList = self.nothingImages

    # Option to ForceAwake pet
    def sleep_test(self):
        if self.sleeping:
            result = (QtGui.QMessageBox.warning(self, "WARNING", "He gets grouchy if he doesn't get his beauty sleep.\
                                                                Are you sure you want to disturb him?",
                                                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
                                                QtGui.QMessageBox.No))
            if result == QtGui.QMessageBox.Yes:
                self.sleeping = False
                self.happiness -= 4
                self.forceAwake = True
                return True
            else:
                return False
        else:
            return True

    # clicking feed button
    def feed_Click(self):
        if self.sleep_test():
            self.imageList = self.eatImages
            self.eating = True
            self.walking = False
            self.playing = False
            self.doctor = False

    # clicking the walk button
    def walk_Click(self):
        if self.sleep_test():
            self.imageList = self.walkImages
            self.eating = False
            self.walking = True
            self.playing = False
            self.doctor = False

    # clicking the play button
    def play_Click(self):
        if self.sleep_test():
            self.imageList = self.playImages
            self.eating = False
            self.walking = False
            self.playing = True
            self.doctor = False

    # clicking the doctor button
    def doctor_Click(self):
        if self.sleep_test():
            self.imageList = self.doctorImages
            self.eating = False
            self.walking = False
            self.playing = False
            self.doctor = True

    # renaming pet
    def rename_Click(self):
        self.name = easygui.enterbox("What do you want to rename your pet to?")
        self.petName.setText(self.name)


    #end
    def stop_Click(self):
        if not self.sleeping:
            self.imageList = self.nothingImages
            self.eating = False
            self.walking = False
            self.playing = False
            self.doctor = False

    # pause
    def pause_Click(self, event):
        self.pause = not self.pause
        if self.pause:
            self.actionPause.setText("Resume")
            self.actionPause.setIcon(QtGui.QIcon("resumeButton.gif"))
        else:
            self.actionPause.setText("Pause")
            self.actionPause.setIcon(QtGui.QIcon("pauseButton.gif"))


    # animation to cycle through images
    def animation_timer(self):
        if self.sleeping and not self.forceAwake:
            self.imageList = self.sleepImages
        self.imageIndex += 1
        if self.imageIndex >= len(self.imageList):
            self.imageIndex = 0
        icon = QtGui.QIcon()

        # update/animate pets image
        current_image = self.imageList[self.imageIndex]
        icon.addPixmap(QtGui.QPixmap(current_image),
                       QtGui.QIcon.Disabled,
                       QtGui.QIcon.Off)
        self.petPic.setIcon(icon)

        # Progress Bar where PB_1 = hunger, PB_2 = Happiness, PB_3 = Health
        self.progressBar_1.setProperty("value", (8 - self.hunger) * (100 / 8.0))
        self.progressBar_2.setProperty("value", self.happiness * (100 / 8.0))
        self.progressBar_3.setProperty("value", self.health * (100 / 8.0))

    # running during the game
    def tick_timer(self):
        if not self.pause:
            self.time_cycle += 1
            if self.time_cycle == 60:
                self.time_cycle = 0
            if self.time_cycle <= 48 or self.forceAwake:
                self.sleeping = False
            else:
                self.sleeping = True
            if self.time_cycle == 0:
                self.forceAwake = False
            if self.doctor:
                self.health += 1
                self.hunger += 1
            elif self.walking and (self.time_cycle % 2 == 0):
                self.happiness += 1
                self.health += 1
                self.hunger += 1
            elif self.playing:
                self.happiness += 2
                self.hunger += 1
            elif self.eating:
                self.hunger -= 2
            elif self.sleeping:
                if self.time_cycle % 3 == 0:
                    self.hunger += 1
            else:
                self.hunger += 1
                if self.time_cycle % 2 == 0:
                    self.happiness -= 1

            # setting conditions if factors are met
            if self.hunger > 8:  self.hunger = 8
            if self.hunger < 0:  self.hunger = 0
            if self.hunger == 7 and (self.time_cycle % 2 == 0):
                self.health -= 1
            if self.hunger == 8:
                self.health -= 1
            if self.health > 8:  self.health = 8
            if self.health < 0:  self.health = 0
            if self.happiness > 8:  self.happiness = 8
            if self.happiness < 0:  self.happiness = 0
            #update progress bars
            self.progressBar_1.setProperty("value", (8 - self.hunger) * (100 / 8.0))
            self.progressBar_2.setProperty("value", self.happiness * (100 / 8.0))
            self.progressBar_3.setProperty("value", self.health * (100 / 8.0))



# saving game save
def closeEvent(self, event):
    file = open("savedata_vp.pkl", 'w')
    save_list = [self.happiness,
                 self.health,
                 self.hunger,
                 datetime.datetime.now(),
                 self.time_cycle,
                 self.name,
                 self.pause]
    pickle.dump(save_list, file)
    event.accept()


def menuExit_selected(self):
    self.close()


app = QtGui.QApplication(sys.argv)
myapp = MyPet()
myapp.show()
app.exec_()
