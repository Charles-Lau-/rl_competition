import random
import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random

class helicopter_agent(Agent):
	randGenerator=Random()
	lastAction= None
	lastObservation= None
	rangeAction = None
	rangeObservation = None
	discountFactor = 1

      	#for approximate reward function 
	reward_list = []
	observation_list = []
	reward_weight = []
        # end

	#for approximate value function
	last_observation_list = []
	action_list = []
	next_observation_list = []
	value_function_weight = []
	#end

	def agent_init(self,taskSpecString):
		"""
			obtain range of observation , range of aciont and discount factor

		"""
		TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString) 
		if(TaskSpec.valid):
			self.discountFactor = TaskSpec.getDiscountFactor()
			self.rangeObservation = TaskSpec.getDoubleObservations()
			self.rangeAction = TaskSpec.getDoubleActions()
		else:
			print "Task Spec could not be parsed: "+taskSpecString;
		
		 

	def agent_start(self,observation):
		#Generate random action, 0 or 1  
		self.lastAction = None
		self.lastObservation = None
		thisDoubleAction=self.randomAction()
		 		
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		#test for approximate value function		
			
 		self.last_observation_list.append(observation.doubleArray)
		self.action_list.append(thisDoubleAction)

		#end of test
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		 
		return returnAction
	
	def agent_step(self,reward, observation):
		#Generate random action, 0 or 1
		
		#approximate value function		
		self.next_observation_list.append(observation.doubleArray)               
		self.approximateValueFunction()		
		#end of test
		thisDoubleAction=self.randomAction()  
		print reward, observation.doubleArray
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		 

		#approximate value function
		self.action_list.append(thisDoubleAction)
		self.last_observation_list.append(observation.doubleArray)
		#end of test


		#test how reward approximation works
		self.approximateRewardFunction(reward,observation)
		#end of test
		
		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		
		 
		return returnAction
	
	def agent_end(self,reward): 
		pass

	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		pass
 	
	def approximateValueFunction(self):
		"""
			try to approximate value function: V(s) = A*s+B*a


		"""
		if(len(self.action_list)<2):
			return
		import numpy
		import math
		Coff = numpy.linalg.lstsq([self.last_observation_list[i]+self.action_list[i] for i in range(0,len(self.action_list))],self.next_observation_list)
		
		#get the error
		total_error = 0
		for i in range(0,len(self.action_list)):
			value = 0
			combined_list = self.last_observation_list[i] + self.action[j]
			for j in range(0,16):
				value += Coff[j]*combined_list[j]
			total_error += math.pow(value-next_observation)
				
	def approximateRewardFunction(self,reward,observation):
		"""	
			try to approximate reward function r = A*s  

		"""
		self.reward_list.append([reward])
		self.observation_list.append(observation.doubleArray)

		if(len(self.reward_list)<2):
			return
		
		import numpy
		import math
		A = numpy.linalg.lstsq(self.observation_list,self.reward_list)[0]
		
 		#get the error
		total_error =0
		for i in range(0,len(self.reward_list)):
			value = 0
			for j in range(0,len(A)):
				value += A[j]*self.observation_list[i][j]
			total_error +=math.pow(value-self.reward_list[i][0],2)  

		 
	def randomAction(self):
		"""
			generate random action.--- test purpose

		"""
		if(self.lastAction==None):
			action = []
			action_length = len(self.rangeAction) 
			for i in range(0,action_length):
				action.append(self.randGenerator.uniform(self.rangeAction[i][0],self.rangeAction[i][1]))		
		
		
			return action
		else: 
			 	 
			return [-i for i in  self.lastAction.doubleArray]		

if __name__=="__main__":
	AgentLoader.loadAgent(helicopter_agent())
