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
	lastAction=Action()
	lastObservation=Observation()
	rangeAction = None
	rangeObservation = None
	discountFactor = 1
      
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
		print observation.doubleArray
		thisDoubleAction=self.randomAction()
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		
		lastAction=copy.deepcopy(returnAction)
		lastObservation=copy.deepcopy(observation)
		 
		return returnAction
	
	def agent_step(self,reward, observation):
		#Generate random action, 0 or 1
		print observation.doubleArray 
		thisDoubleAction=self.randomAction()
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		
		lastAction=copy.deepcopy(returnAction)
		lastObservation=copy.deepcopy(observation)
		
		print returnAction.doubleArray
		return returnAction
	
	def agent_end(self,reward):
		pass

	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		pass
	
	def randomAction(self):
		"""
			generate random action.--- test purpose

		"""
		action = []
		action_length = len(self.rangeAction) 
		for i in range(0,action_length):
			action.append(self.randGenerator.uniform(self.rangeAction[i][0],self.rangeAction[i][1]))		
		
		return action
		

if __name__=="__main__":
	AgentLoader.loadAgent(helicopter_agent())
