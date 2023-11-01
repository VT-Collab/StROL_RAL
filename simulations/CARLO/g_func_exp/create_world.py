import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting, LineDraw
from geometry import Point, Line


class Create_World:

	def __init__(self):
		self.dt = None
		self.w = None
		self.c1 = None
		self.p1 = None
		self.predPed = None
		self.initPed = None

		# self.Intersection()

	def Intersection(self):
		self.dt = 0.1
		self.w = World(self.dt, width=100, height=100, ppm=5)
		
		# line that points from pedestrian to the estimated goal
		self.predPed = LineDraw(Point(0, 0), Point(0, 0), 'red')
		self.w.add(self.predPed)

		## ADD PARKS AND SIDEWALKS

		# Bottom Left
		self.w.add(Painting(Point(20, 20), Point(42, 42), 'gray80'))
		self.w.add(Painting(Point(17, 17), Point(35, 35), 'green'))
		# self.w.add(RectangleBuilding(Point(24, 24), Point(49, 49)))

		# Bottom Right
		self.w.add(Painting(Point(80, 20), Point(42, 42), 'gray80'))
		self.w.add(Painting(Point(82, 17), Point(35, 35), 'green'))
		# self.w.add(RectangleBuilding(Point(96, 24), Point(35, 35)))

		# Top Left
		self.w.add(Painting(Point(20, 80), Point(42, 42), 'gray80'))
		self.w.add(Painting(Point(17, 82), Point(35, 35), 'green'))
		# self.w.add(RectangleBuilding(Point(24, 96), Point(35, 35)))

		# Top Right 
		self.w.add(Painting(Point(80, 80), Point(42, 42), 'gray80'))
		self.w.add(Painting(Point(82, 82), Point(35, 35), 'green'))
		# self.w.add(RectangleBuilding(Point(96, 96), Point(35, 35)))

		## ADD CROSSWALKS
		
		# Left
		self.w.add(Painting(Point(38, 54), Point(6.66, 2.3), 'white'))
		self.w.add(Painting(Point(38, 51), Point(6.66, 2.3), 'white'))
		self.w.add(Painting(Point(38, 48), Point(6.66, 2.3), 'white'))
		self.w.add(Painting(Point(38, 45), Point(6.66, 2.3), 'white'))

		# Right
		self.w.add(Painting(Point(62, 54), Point(6.66, 2.3), 'white'))
		self.w.add(Painting(Point(62, 51), Point(6.66, 2.3), 'white'))
		self.w.add(Painting(Point(62, 48), Point(6.66, 2.3), 'white'))
		self.w.add(Painting(Point(62, 45), Point(6.66, 2.3), 'white'))

		# Top
		self.w.add(Painting(Point(45, 62), Point(2.3, 6.66), 'white'))
		self.w.add(Painting(Point(48, 62), Point(2.3, 6.66), 'white'))
		self.w.add(Painting(Point(51, 62), Point(2.3, 6.66), 'white'))
		self.w.add(Painting(Point(54, 62), Point(2.3, 6.66), 'white'))

		# Bottom
		self.w.add(Painting(Point(45, 38), Point(2.3, 6.66), 'white'))
		self.w.add(Painting(Point(48, 38), Point(2.3, 6.66), 'white'))
		self.w.add(Painting(Point(51, 38), Point(2.3, 6.66), 'white'))
		self.w.add(Painting(Point(54, 38), Point(2.3, 6.66), 'white'))

		## ADD ROBOT CAR
		self.c1 = Car(Point(50, 100-90), 1*np.pi/2, 'orange')
		self.c1.velocity = Point(0.0, 5)
		self.c1.max_speed = 20.0
		self.c1.min_speed = 0.0
		self.c1.acceleration = 0.10
		self.w.add(self.c1)
		
		self.peds = []
		
		# generated random pedestrians
		x = 0 + 15*np.random.rand();
		y = 100 - 15*np.random.rand();
		self.p1 = Pedestrian(Point(x, y) , np.pi, 'blue')
		self.p1.velocity = Point(0.0, 0.0)
		self.p1.max_speed = 0.0
		self.p1.min_speed = 0.0
		self.p1.set_control(0, 0)
		self.w.add(self.p1)
		
		## ADD PEDESTRIAN GOALS
		self.g1 = Painting(Point(40, 30), Point(1.5, 1.5), 'red')
		
		self.g3 = Painting(Point(70, 60), Point(1.5, 1.5), 'red')

		self.GOALS = np.array([self.g3, self.g1])
		for g in self.GOALS:
			self.w.add(g)

		self.goal_r = Painting(Point(50, 100-4), Point(4, 4), 'yellow')
		self.w.add(self.goal_r)

		self.w.render()

		return self.w