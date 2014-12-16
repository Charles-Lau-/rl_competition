import math
import numpy 
import random

def randomGaussianAction(lastAction):
		"""
			use gaussian distribution to generate random action smoothly.--- test purpose

		"""
		 
		
		if(lastAction==None):
			return [0.0,0.0,0.0,0.0]
		
		mean =  lastAction
		cov = [ [0]*4 for i in range(0,4)]
		
		#the cov should be tuned
		for i in range(0,4):
			cov[i][i] = 0.001
  		
		action = numpy.random.multivariate_normal(mean,cov)  
		return action


def randomSmoothAction(lastAction):
		"""
			generate random action smoothly.--- test purpose

		"""
		 
		if(lastAction==None):
			return [0.0,0.0,0.0,0.0]
		action = []
		action_length = 4 
		for i in range(0,action_length):
			sign  = random.uniform(0,1)
			if(sign < 0.5):
				action.append(math.pow(random.uniform(0,0.15)+lastAction[i],2)+random.uniform(0,0.1))		
			else:
				action.append(math.pow(lastAction[i]-random.uniform(0,0.15),2)-random.uniform(0,0.1))		
		 		  
		return action	 	

def randomAction():
		"""
			generate random action.--- test purpose

		"""
		 
		action = []
		for i in range(0,4):
			action.append(random.uniform(-1,1))

		return action
