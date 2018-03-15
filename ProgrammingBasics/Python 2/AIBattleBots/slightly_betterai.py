class AI:
    def __init__(self):
        pass
    def turn(self):
        if self.robot.lookInFront() == "bot":
            self.robot.attack()
        else:
            self.goTowards(self.robot.locateEnemy()[0])

    def goTowards(self,enemyLocation):
        mylocation = self.robot.position
        changeInDistance = (enemyLocation[0]-mylocation[0],
                            enemyLocation[1]-mylocation[1])
        if abs(changeInDistance[0]) > abs(changeInDistance[1]):
            if changeInDistance < 0:
                #face west
                targetOrientation = 3
            else:
                #face east
                targetOrientation = 1
        else:
            if changeInDistance[1] < 0:
                #north
                targetOrientation = 0
            else:
                #south
                targetOrientation = 2
        #Onwards my noble steed!
        if self.robot.rotation == targetOrientation:
            self.robot.goForth()
        else:
            leftTurnsNeeded = (self.robot.rotation - targetOrientation) % 4
            if leftTurnsNeeded <=2:
                self.robot.turnLeft()
            else:
                self.robot.turnRight()