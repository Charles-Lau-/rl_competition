import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
import numpy
import util

class helicopter_agent(Agent):
	randGenerator=Random()
	lastAction= None
	lastObservation= None
	rangeAction = None
	rangeObservation = None
	discountFactor = 1

	transition_func = None
       	reward_func = None
	dataset = None
	#measure performance
	episode_counter = 0
	steps = 0
	reward = 0
	
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
		
		#initial a network [12,12,4] and initial weights are baseline policy versions
		 
		self.transition_func = StateActionToStateNetwork()
		self.reward_func = StateToRewardNetwork()

	def agent_start(self,observation):
		#Generate random action, 0 or 1  
		self.lastAction = None
		self.lastObservation = None
		self.episode_counter += 1
		self.steps = 0
		self.reward  = 0
		  
		thisDoubleAction=self.agent_step_action(observation.doubleArray)
		
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
	 

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		 
		return returnAction
	
	def agent_step(self,reward, observation): 
		 		
		
		print reward
		 
		thisDoubleAction=self.agent_step_action(observation.doubleArray)  
		  
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		 
		 
		self.lastObservation=copy.deepcopy(observation)
		self.lastAction=copy.deepcopy(returnAction)
		
		 
		return returnAction
	
	def agent_step_action(self,observation):			
		 
 
		return self.network.activate(observation)

		
	def agent_end(self,reward):
		pass
	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		pass
 	 
						

	 
def StateActionToStateNetwork():
	#initial a network [16,12] 

	from pybrain.structure import FeedForwardNetwork,LinearLayer,TanhLayer,FullConnection
	network = FeedForwardNetwork()
	inLayer= LinearLayer(16)
	hiddenLayer = TanhLayer(16)
	outLayer = TanhLayer(12)
	network.addInputModule(inLayer)
	network.addModule(hiddenLayer)
	network.addOutputModule(outLayer)
	in_to_hidden = FullConnection(inLayer,hiddenLayer) 
	hidden_to_out = FullConnection(hiddenLayer,outLayer) 		
	network.addConnection(in_to_hidden)
	network.addConnection(hidden_to_out)
	network.sortModules()
	return network 

def StateToRewardNetwork():
	#initial a network [12,1] 

	from pybrain.structure import FeedForwardNetwork,LinearLayer,TanhLayer,FullConnection
	network = FeedForwardNetwork()
	inLayer= LinearLayer(12)
	hiddenLayer = TanhLayer(12)
	outLayer = TanhLayer(1)
	network.addInputModule(inLayer)
	network.addModule(hiddenLayer)
	network.addOutputModule(outLayer)
	in_to_hidden = FullConnection(inLayer,hiddenLayer) 
	hidden_to_out = FullConnection(hiddenLayer,outLayer) 		
	network.addConnection(in_to_hidden)
	network.addConnection(hidden_to_out)
	network.sortModules()
	return network 
 
 
 

if __name__=="__main__":
	AgentLoader.loadAgent(helicopter_agent())
