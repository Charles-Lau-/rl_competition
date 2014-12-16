import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
import numpy
import util #module written by myself,used for randomAction generation

 
"""
	step_size need to be reconsidered
	how to generate random action it need to be considered thoroughly
	how to set epsilon

	these three factors will affect the convergence and performance
"""
class helicopter_agent(Agent):
	randGenerator=Random()
	lastAction= None
	lastObservation= None
	rangeAction = None
	rangeObservation = None
	discountFactor = 1
       	j = 0
	step_size = 0.05
	#Q-table : Q(S) = {"action 1":reward,"action 2":reward...}
	Q_Table = {}
	 
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
		
		#initialization of Q_table 
		self.Q_Table = {str([0.0]*12):{str([0.0]*4):0}}	 
		
	def agent_start(self,observation):
		self.j +=1

		self.lastAction = None
		self.lastObservation = None

		print " " 
		thisDoubleAction=self.agent_action_start(observation.doubleArray)
		
		if(self.j>10000):
			print thisDoubleAction 

		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
	  

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		 
		return returnAction
	
	def agent_step(self,reward, observation):
		
		 		
		
		print reward 
		thisDoubleAction=self.agent_action_step(reward,observation.doubleArray)  
		  
		
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		 
		
	 	 

		self.lastObservation=copy.deepcopy(observation)
		self.lastAction=copy.deepcopy(returnAction)
		
		 
		return returnAction
	
	def agent_end(self,reward): 
		"""
			remove last observation from the table

		"""	
		key  = self.discretize_observation(self.lastObservation.doubleArray)
		self.Q_Table.pop(str(key))		

	def agent_cleanup(self):
		t = open("data","w")
		for key,value in self.Q_Table.items():
			t.write(key)
			t.write("\r\n")
			for k,v in value.items():
				t.writelines(k)
				t.write("=====")
				t.write(str(v))
				t.write("\r\n")
			t.write("\r\n")
			t.write("\r\n")
			t.write("===============================")
			t.write("\r\n")
		t.close()

	def agent_message(self,inMessage):
		pass

	#take action 	 
	def agent_action_step(self,reward,observation):
		discretized_observation =  self.discretize_observation(self.lastObservation.doubleArray)
		discretized_action  = self.discretize_action(self.lastAction.doubleArray)
		discretized_next_observation = self.discretize_observation(observation)
		 
		next_key = str(discretized_next_observation)
		obs_key = str(discretized_observation)
		act_key = str(discretized_action)		
		#get Q-value of last state and last action	
		actions = self.Q_Table[obs_key]
		if(actions.has_key(act_key)):
			Q_s_a = actions[act_key]		
		else:
			Q_s_a = 0 
	 
		#get max Q-value of next state
		if(self.Q_Table.has_key(next_key)):
			cand_actions = self.Q_Table[next_key]
			 
			action_with_max_value = max(cand_actions,key = cand_actions.get)
			Q_next_max = cand_actions[action_with_max_value]
								
		else:
			Q_next_max = 0
		
		#update the value
		Q_s_a = Q_s_a +	self.step_size*(reward + Q_next_max - Q_s_a)
		self.Q_Table[obs_key][act_key] = Q_s_a 
		 
		#generate new action 
		if(self.Q_Table.has_key(next_key)):
			if(self.randGenerator.uniform(0,1)> (0.7 / (self.j - 10000 > 0 and (self.j-10000)/10.0 or 1.0)) ):
				cand_actions = self.Q_Table[next_key] 
				return [float(i[1:]) for i in max(cand_actions,key =  cand_actions.get).split("]")[0].split(",")]			
			else:
				return util.randomGaussianAction(self.lastAction.doubleArray)	 	 				
		else:
			self.Q_Table[next_key] = {}
			return util.randomGaussianAction(self.lastAction.doubleArray)		
		 
	def agent_action_start(self,observation):
		observation = [i for i in observation]
		actions = self.Q_Table[str(observation)]
		desired_action = None

		related_reward = -1000				
		if(self.randGenerator.uniform(0,1)> (0.7 / (self.j - 10000 > 0 and (self.j-10000)/10.0 or 1.0))):
			desired_action = max(actions,key=actions.get)		
			return [float(i[1:]) for i in copy.copy(desired_action).split("]")[0].split(",")]
		else:
			return util.randomGaussianAction([0.0,0.0,0.0,0.0])	
						 		
	#two function used to discretize search space			
  	def discretize_action(self,action):
		"""
			keep x.1  form of action

		"""
		 
		result = []
		for i in action:
			if(float("%.4f" % i) == -0.0):
				result.append(0.0)
			else:
				result.append(float("%.4f" % i))
		return result

	def discretize_observation(self,observation):
		"""
			keep xx.5 xx.0 form of observation
		"""
		
		result = []
		for i in observation:
			if(float("%.1f" % i) == -0.0):
				result.append(0.0)
			else:
				result.append(float("%.1f" % i))
		return result

if __name__=="__main__":
	AgentLoader.loadAgent(helicopter_agent())
