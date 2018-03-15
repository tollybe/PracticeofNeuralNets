from psychopy import visual, core, event
import random

numberOfTrials = 200 # number of trials to be tested
validity = 0.75 # accuracy of the arrow pointing
fixTime = 0.25 # time "+" is shown
arrowTime = 0.5 # arrow time shown

rt = [] #reaction time results
tn = [] #trial number (i)
cueside = [] #where arrows show
valid = [] #number of answers
correct = []

targSide = "z"
fix = "+"
arrows = ["<",">"]
targetSymbol = "#"

#creating window (expWin) and stimuli ( where fixspot =+, arrows = <, target = #)
window = visual.Window(size=(400,400),fullscr=1)
fixSpot = visual.TextStim(window,pos=(0,0),text= fix, color= [255,0,0]) # centering +
leftArrow = visual.TextStim(window, pos = (0,0), text= arrows[0], height = 0.2) # centering <<<
rightArrow = visual.TextStim(window, pos = (0,0), text = arrows[1], height = 0.2) # centering >>>
target = visual.TextStim(window,pos= (0,0), text= targetSymbol, height= 0.4)

#creating timer
timer = core.Clock()

#simulating a window that iterates through the trial lengths
for i in range(1,numberOfTrials+1):
    """
    draws fix spot, flips window, waits for time, and simulates arrow
    """
    fixSpot.draw()
    window.flip()
    core.wait(fixTime)

    """
    Deciding whether the position of the target, and recording it on "cueside"
    """
    if random.random() <0.5:
        leftArrow.draw()
        cueside.append("L")
        targetPositioning = (-.5,0)
    else:
        rightArrow.draw()
        cueside.append("R")
        targetPositioning = (.5,0)
        targSide = "slash"
    window.flip()
    tn.append(i)
    core.wait(arrowTime)

    """
    assessing whether the position is accurate (e.g. < = z or < = /)
    """
    if random.random() <validity:
        valid.append("T")
    else:
        valid.append("F")
        targetPositioning -(targetPositioning[0]*(-1),targetPositioning[1])
        if cueside [-1] == "L":
            targSide = "slash"
        else:
            targSide = "z"

    target.setPos(targetPositioning)
    target.draw()
    window.flip()
    timer.reset()
    buttonPress = event.waitKeys() #detecting a button press that indicates whether participant detected target
    print(buttonPress)
    rt.append(timer.getTime()) #record time
    """
    whether the appropriate button press is correct and appending to "correct" score as T, or F
    """
    if (valid[-1] == "T"):
        if (buttonPress[-1] =="slash" and cueside =="R" or (buttonPress[-1] =="z" and cueside =="L")):
            correct.append("T")
        else:
            correct.append("F")
    else:
        if(buttonPress[-1] == "slash" and cueside =="R" or (buttonPress[-1] == "z" and cueside =="L")):
            correct.append("F")
        else:
            correct.append("T")

"""
Storing this on a text file detailing: trial number, Cue position, valid position, reaction time, and number of correct
scores.
"""
f = open("./posnerFata.txt","w")
f.write("TN\tCue\tValid\tReaction Time\tCorrect\n")
for i in range(0,numberOfTrials):
    f.write(
            str(tn["i"])+ "\t" \
            + cueside[i] + "\t" \
            + valid[i] + "\t" \
            + str(rt[i] + "\t")\
            + correct[i] + "\t"
            )
f.close()

#closing psychopy
core.quit()