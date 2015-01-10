import math
import numpy 
import random



def randomSigmoidEpsilon(reward,a,b):
		"""
			epsilon =  lambda r,a,b: 1.0/(1+math.exp(a*(r+b))) 

		"""

		return 1.0/(1+math.exp(a*(reward+b)))

def randomExponenEpsilon(reward,a):
		"""
			exponen_fun = lambda r,a: a*math.exp(-r*math.log(1.0/a)/100.0)
		"""
		
		return a*math.exp(-r*math.log(1.0/a/100.0))

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
			cov[i][i] = 0.002
  		
		action = numpy.random.multivariate_normal(mean,cov)  
		return action

def evolutionMutate():
	"""
		serve for the evolution algorithm , use 0 as mean and 0.001 as variance to generate random value

	"""
	return random.gauss(0,0.01)

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



def baselinePolicy(state):
	"""
		given a state, a baseline policy is used to determine the next action


	"""
	weights = [0.0196, 0.7475, 0.0367, 0.0185,0.7904, 0.0322, 0.1969, 0.0513, 0.1348, 0.02, 0, 0.23]
	
	action = [0]*4 
	action[0] = -weights[0]*state[4]+ \
		    -weights[2]*state[1]+ \
		    -weights[1]*state[9]+ \
		    weights[9]

	action[1] = -weights[3]*state[3]+ \
		    -weights[5]*state[0]+ \
		    weights[4]*state[10]+ \
		    weights[10]
	action[2] = -weights[6]*state[11]
	action[3] =  weights[7]*state[5]+ \
		     weights[8]*state[2]+ \
		     weights[11]

	return action




