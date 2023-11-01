import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
from interactive_controllers import KeyboardController

class RobotCar(Car):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_speed = 2.0
        self.min_speed = -2.0
        self.velocity = Point(0.0, 0.0)
        self.goal = kwargs.get('goal', None)
        self.goal_reached = False
        self.goal_reached_time = 0.0
        self.collided = False
        self.score = 0.0
        self.theta_star = kwargs.get('theta_star', None)
        self.prevHumanSteps = []
    
    def calcThetaStar(self, goals):
        # calculate theta_star, which is the predicted goal of the human pedestrian
        # based on the previous human steps
        theta_star = goals[0] #calculated from predictive model
        return theta_star
    
    def updateHumanSteps(self, dt, human):
        # update the previous human steps
        self.prevHumanSteps.append((dt, human.position))
    
    
    def predictHumanTraj(self, goals, dt):
        # predict the human trajectory based on the previous human steps
        # if the human's trajectory crosses the robot's goal, and the robot could 
        # hit the human, then the robot should slow down. Otherwise, the robot
        # should speed up or maintain its current speed.
        self.theta_star = self.calcThetaStar(goals)
        
        curHumanPos = self.prevHumanSteps[-1][1]
        if (self.position.x < curHumanPos.x and self.theta_star.x < self.position.x) or (self.position.x > curHumanPos.x and self.theta_star.x > self.position.x):
            # the human's trajectory will cross the robot's goal           
            curHumanVel = (curHumanPos - self.prevHumanSteps[-2][1]) / self.prevHumanSteps[-1][0]
            m = (self.theta_star.y - curHumanPos.y) / (self.theta_star.x - curHumanPos.x)
            b = self.theta_star.y - m * self.theta_star.x
            intersection = (self.goal.x, m * self.goal.x + b)
            robotVel = self.velocity
            robotAccel = self.acceleration
            distToIntersectHuman = np.sqrt((intersection[0] - curHumanPos.x)**2 + (intersection[1] - curHumanPos.y)**2)
            distToIntersectRobot = np.sqrt((intersection[0] - self.position.x)**2 + (intersection[1] - self.position.y)**2)
            timeToIntersectHuman = distToIntersectHuman / curHumanVel
            # determine if the robot will hit the human at the current speed and acceleration
            
        else:
            return None
        


class Create_World:

    def __init__(self):
        self.dt = None
        self.w = None
        self.c1 = None
        self.p1 = None
        
    def Intersection(self):
        self.dt = 0.1
        self.w = World(self.dt, width=120, height=120, ppm=5)

        ## ADD PARKS AND SIDEWALKS

        # Bottom Left
        self.w.add(Painting(Point(25, 25), Point(51, 51), 'gray80'))
        self.w.add(Painting(Point(21, 21), Point(43, 43), 'green'))
        # self.w.add(RectangleBuilding(Point(24, 24), Point(49, 49)))

        # Bottom Right
        self.w.add(Painting(Point(95, 25), Point(51, 51), 'gray80'))
        self.w.add(Painting(Point(99, 21), Point(43, 43), 'green'))
        # self.w.add(RectangleBuilding(Point(96, 24), Point(43, 43)))

        # Top Left
        self.w.add(Painting(Point(25, 95), Point(51, 51), 'gray80'))
        self.w.add(Painting(Point(21, 99), Point(43, 43), 'green'))
        # self.w.add(RectangleBuilding(Point(24, 96), Point(43, 43)))

        # Top Right 
        self.w.add(Painting(Point(95, 95), Point(51, 51), 'gray80'))
        self.w.add(Painting(Point(99, 99), Point(43, 43), 'green'))
        # self.w.add(RectangleBuilding(Point(96, 96), Point(43, 43)))

        ## ADD CROSSWALKS
        
        # Left
        self.w.add(Painting(Point(45.5, 66), Point(8, 2.8), 'white'))
        self.w.add(Painting(Point(45.5, 62), Point(8, 2.8), 'white'))
        self.w.add(Painting(Point(45.5, 58), Point(8, 2.8), 'white'))
        self.w.add(Painting(Point(45.5, 54), Point(8, 2.8), 'white'))

        # Right
        self.w.add(Painting(Point(74.5, 66), Point(8, 2.8), 'white'))
        self.w.add(Painting(Point(74.5, 62), Point(8, 2.8), 'white'))
        self.w.add(Painting(Point(74.5, 58), Point(8, 2.8), 'white'))
        self.w.add(Painting(Point(74.5, 54), Point(8, 2.8), 'white'))

        # Top
        self.w.add(Painting(Point(54, 74.5,), Point(2.8, 8), 'white'))
        self.w.add(Painting(Point(58, 74.5,), Point(2.8, 8), 'white'))
        self.w.add(Painting(Point(62, 74.5,), Point(2.8, 8), 'white'))
        self.w.add(Painting(Point(66, 74.5,), Point(2.8, 8), 'white'))

        # Bottom
        self.w.add(Painting(Point(54, 45.5,), Point(2.8, 8), 'white'))
        self.w.add(Painting(Point(58, 45.5,), Point(2.8, 8), 'white'))
        self.w.add(Painting(Point(62, 45.5,), Point(2.8, 8), 'white'))
        self.w.add(Painting(Point(66, 45.5,), Point(2.8, 8), 'white'))
        
        peds = []
        
        # generated random pedestrians
        x = np.random.rand()*120;
        y = np.random.rand()*120;
        while ((x>=51 and x<=68) and ((y>=51 and y<=68))) or ((x>=51 and x<=68) and ((y >= 77 and y<=120) or (y>=0 and y<=43))) or (((x >=77 and x <= 120 ) or (x>=0 and x<=43)) and ((y>=51 and y<=68))):
            x = np.random.rand()*120;
            y = np.random.rand()*120;
        peds.append(Pedestrian(Point(x, y), np.pi, 'blue'))
        
        for p in peds:
            p.velocity = Point(0.0, 0.0)
            p.max_speed = 1.0
            p.min_speed = 0.0
            self.w.add(p)
        
        ## ADD PEDESTRIAN GOALS
        self.g1 = Painting(Point(20, 47.5), Point(1.5, 1.5), 'red')
        self.w.add(self.g1)

        self.g2 = Painting(Point(72.5, 20), Point(1.5, 1.5), 'red')
        self.w.add(self.g2)

        self.g3 = Painting(Point(72.5, 100), Point(1.5, 1.5), 'red')
        self.w.add(self.g3)
        
        self.GOALS = np.array([self.g1, self.g2, self.g3])
        
        self.goal_r = Painting(Point(65, 115), Point(4, 4), 'red')
        self.w.add(self.goal_r)

        self.w.render()

        return self.w



    