import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
import numpy

#module written by myself,used for randomAction generation
import util 
 
"""
Tricks can be done for performance improvement:

	Generate random action more smart
	set learning rate epsilon in a considerable way
	to avoid bad action and bad state

	these three factors will affect the convergence and performance

"""

class helicopter_agent(Agent):
	randGenerator=Random()
	lastAction= None
	lastObservation= None
	rangeAction = None
	rangeObservation = None
	discountFactor = 1
       
	 
	#Q-table : Q(S) = {"action 1":reward,"action 2":reward...}
	Q_Table = {}
	Bad_Table = {}
 	Step_Size = {}
	#learning rate
	Epsilon = 0.2 

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
		
		self.Episode_Counter +=1

		self.lastAction = None
		self.lastObservation = None

		print " " 
		thisDoubleAction=self.agent_action_start(observation.doubleArray)
	 

		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
	  

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		 
		return returnAction

	def isInBlackList(self,observation,action):
		"""
			check whether generated (ob,a) is in the black list , if so re-generate this pair

		"""		
		if(self.Bad_Table.has_key(observation) and self.BadTable[observation].has_key(action)):
			return True
	
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
			update on termination state  

		"""	
		observation_key  = convertListToString(self.discretize_observation(self.lastObservation.doubleArray))
		action = self.discretize_action(self.lastAction.doubleArray)		
		action_key = convertListToString(action)
				
		#update Q_Table
		self.Q_Table[observation_key][action_key] = reward

		#update Bad_Table	
		self.Bad_Table.has_key(observation_key) and \
			self.Bad_Table[observation_key].append(action) or \
				self.Bad_Table.setdefault(observation_key,[action])

	def agent_cleanup(self):
		"""
			persist data			

		"""
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
		"""
			dealing with Td-0 procedure

		"""
		discretized_observation =  self.discretize_observation(self.lastObservation.doubleArray)
		discretized_action  = self.discretize_action(self.lastAction.doubleArray)
		discretized_next_observation = self.discretize_observation(observation)
		 
		next_key = convertListToString(discretized_next_observation)
		obs_key = convertListToString(discretized_observation)
		act_key = convertListToString(discretized_action)		
		
		#get Q-value of last state and last action	
		actions = self.Q_Table[obs_key]
		Q_s_a   = actions.has_key(act_key) and \
			   actions[act_key] or \
		   	   self.Q_Table[obs_key].setdefault(act_key,0) 
	 
		#get max Q-value of next state
		if(self.Q_Table.has_key(next_key)):
			cand_actions = self.Q_Table[next_key]			 
			action_with_max_value = max(cand_actions,key = cand_actions.get)
			Q_next_max = cand_actions[action_with_max_value]
								
		else:
			Q_next_max = 0 
		#update step_size
		if(self.Step_Size.has_key(obs_key) ):
			self.Step_Size[obs_key].has_key(act_key) and \
				self.Step_Size[obs_key].setdefault(act_key,self.Step_Size[obs_key][act_key]+1) or \
					self.Step_Size[obs_key].setdefault(act_key,1) 
 		else:
			self.Step_Size[obs_key] = {act_key:1}

		Q_s_a = Q_s_a +	1.0/self.Step_Size[obs_key][act_key]*(reward + Q_next_max - Q_s_a)
		self.Q_Table[obs_key][act_key] = Q_s_a 
		 
		#generate new action 
		#with probability epsilon generate random action , with 1-epsilon generate greedy action
		if(self.Q_Table.has_key(next_key)):
			if(self.randGenerator.uniform(0,1) > self.Epsilon):
				cand_actions = self.Q_Table[next_key] 		
				return convertStringToList(max(cand_actions,key =  cand_actions.get))

			else:
				return util.randomGaussianAction(self.lastAction.doubleArray)	 	 				
		else:
			self.Q_Table[next_key] = {}
			return util.randomGaussianAction(self.lastAction.doubleArray)		
		 
	def agent_action_start(self,observation):
		"""
			first step in episode
	
		"""
		actions = self.Q_Table[convertListToString(observation)]
		desired_action = None
		related_reward = -1000			
		#generate new action 
		#with probability epsilon generate random action , with 1-epsilon generate greedy action	
		if(self.randGenerator.uniform(0,1)> self.Epsilon):
			desired_action = max(actions,key=actions.get) 	
			return convertStringToList(desired_action)
		else:
			return util.randomGaussianAction([0.0,0.0,0.0,0.0])	
						 		
	#two function used to discretize search space			
  	def discretize_action(self,action):
		"""
			keep x.1  form of action

		"""
		 
		result = []
		for i in action:
			if(float("%.2f" % i) == -0.0):
				result.append(0.0)
			else:
				result.append(float("%.2f" % i))
		return result

	def discretize_observation(self,observation):
		"""
			keep xx.5 xx.0 form of observation
		"""
		
		result = []
		for i in observation:
			if(i-round(i) > 0.7):
				p = round(i,0)
			elif(i-round(i) < 0.3):
				p = round(i,0)
			else:
				p = round(i,0)+0.5
	                if(p == -0.0):
				result.append(0.0)
			else:
				result.append(p) 
		return result

def convertListToString(targetList):
	"""
		convert [1,2,3,4] to "[1,2,3,4]"

	"""	
	return str([i for i in targetList])

def convertStringToList(targetString):
	"""
		convert "[1,2,3,4]" to [1,2,3,4]

	"""
	return [float(i[1:]) for i in targetString.split("]")[0].split(",")]
if __name__=="__main__":
	AgentLoader.loadAgent(helicopter_agent())
