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

From Observation:
	We could know the reason that it does not run for a long time is that it fall into a local minimum
	There is a local deep hole into which is easy  fall   
"""

class helicopter_agent(Agent):
	randGenerator=Random()
	lastAction= None
	lastObservation= None
	rangeAction = None
	rangeObservation = None
	discountFactor = 1
       
	 
	#Q-table : Q(S) = {"action 1":reward,"action 2":reward...}
	#Bad_table : B(S) = [action1,action2...]
	Q_Table = {} 
	Bad_Table = {}
 	Step_Size = {}
	E_Trace = {}

	#learning rate 
	Episode_Counter = 0 
	Epsilon = 0.3
	lambda_ = 0.4 # Monte Carlo

	#balance performace
	Steps = 0
	Overall_Steps = 0
	
	
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

		self.Steps = 1 
		print " " 
		thisDoubleAction=self.agent_action_start(observation.doubleArray)
	 

		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
	  

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		 
		return returnAction


	#optional function
	def isInBlackList(self,observation,action):
		"""
			check whether generated (ob,a) is in the black list , if so re-generate this pair
			And the idea here is that we calculate the similarity(a1,bad_a) in state s, if the similarity is high then we reject it
			and re-generate an action 
		
			About how to measure "high", whether set a hard line or a soft line which is the average similarity of two bad_actions

		"""		
		if(self.Bad_Table.has_key(observation)):
			 	
			actions = self.Bad_Table[observation]
			threshold = 0.0
			similarity = lambda x,y:  numpy.inner(x,y)/(numpy.linalg.norm(x)*numpy.linalg.norm(y))
			length = len(actions)
			 
			if(length<2 and similarity(action,actions[0]) > 0.9):
				return True

			#calculate average similarity
			for i in range(0,length):
				for j in range(i,length): 
					threshold += similarity(actions[i],actions[j])
		
			threshold /= length*(length-1) / 2
			#if  similarity is higher than the average one			
			targetSimilarity = 0.0	
			for i in range(0,length):
				targetSimilarity += similarity(actions[i],action)
  
			targetSimilarity /= length
			if(targetSimilarity > threshold):
				print similarity(actions[i],action),threshold
				return True
		return False

	def agent_step(self,reward, observation):
		
		import math
	
	 
		if(self.Episode_Counter>5000):
			self.Epsilon = 0		
	 
 		
		self.Steps +=1
		

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


		#meansurement of performance
		if(self.Episode_Counter>5000):
			self.Overall_Steps += self.Steps

	def agent_cleanup(self):
		"""
			make data persistant so that it could be analyzed		

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
		
		#measurement of performance
		print(self.Overall_Steps/100.0)

	def agent_message(self,inMessage):
		pass

	def getQValue(self,obs_key,act_key):
		"""
			get Q-value		

		"""
		if(self.Q_Table[obs_key].has_key(act_key)):
			return self.Q_Table[obs_key][act_key]
		else:
			self.Q_Table[obs_key][act_key] = 0
			return 0

	def getQValueOfNextState(self,obs_key,next_key):
		"""
			return Q_best,action_star and next_action

		"""
		if(self.Q_Table.has_key(next_key)):
			cand_actions = self.Q_Table[next_key]
			action_star = max(cand_actions,key = cand_actions.get)
			Q_best = self.Q_Table[next_key][action_star]			 
			if(self.randGenerator.random() > self.Epsilon):
				next_action = action_star
				 
			else:
				actions =  cand_actions.keys()
				next_action = actions[self.randGenerator.randint(0,len(actions)-1)]					
		else:
			next_action = convertListToString(util.randomGaussianAction(self.lastAction.doubleArray))
			action_star = next_action
			Q_best = 0

		return Q_best,action_star,next_action
	
	def updateEligibility(self,obs_key,act_key):
		"""
			update eligibility trace

		"""
		if(self.E_Trace.has_key(obs_key)):
			if(self.E_Trace[obs_key].has_key(act_key)):
				self.E_Trace[obs_key][act_key] += 1

			else:
				self.E_Trace[obs_key][act_key] = 1

		else:
			self.E_Trace[obs_key] = {act_key:1}

	def updateStepSize(self,obs_key,act_key):
		"""
			update step size		

		"""
		if(self.Step_Size.has_key(obs_key) ):
			if(self.Step_Size[obs_key].has_key(act_key)):
				self.Step_Size[obs_key][act_key] += 1
			else:
				self.Step_Size[obs_key][act_key] = 1 
 		else:
			self.Step_Size[obs_key] = {act_key:1}

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
		Q_s_a   = self.getQValue(obs_key,act_key)

		#get action of next state with policy epsilon , best action  and best Q-value
		
		Q_best,action_star,next_action = self.getQValueOfNextState(obs_key,act_key)
		   		
		#get sigma
		sigma = reward + Q_best-Q_s_a
		#update Eligibility Trace
		self.updateEligibility(obs_key,act_key) 
		
		#update step_sized
		 
		self.updateStepSize(obs_key,act_key)

		#update all (s,a)
		for state in self.Step_Size.keys():
			for action in self.Step_Size[state].keys():
				self.Q_Table[state][action] += 1.0/self.Step_Size[state][action]*sigma*self.E_Trace[state][action]
				 				
				if(action_star == next_action):
					self.E_Trace[state][action] *= self.lambda_
				else:
					self.E_Trace[state][action] = 0		 
		 
		#generate new action 
		#with probability epsilon generate random action , with 1-epsilon generate greedy action
		if(self.Q_Table.has_key(next_key)):
			if(self.randGenerator.random() > self.Epsilon):
				cand_actions = self.Q_Table[next_key]
				target = convertStringToList(max(cand_actions,key =  cand_actions.get))
				 
				return target
			else:
				target = util.randomGaussianAction(self.lastAction.doubleArray)
				
				return target	
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
