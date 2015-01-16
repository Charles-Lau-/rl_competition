import random,numpy,util,sys,copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random

"""
    If set initial weights , neural network topology , way of crossover and mutation rate appropriately ,
  	then the performance can be improved

	"""

population_size = 20
mutation_rate = 0.05
decay_rate = 0.998
variance = 0.01

class genome():	
	def __init__(self,network,fittness=0):
		self.network = network
		self.fittness = fittness
	@staticmethod 
	def crossover(genomeOne,genomeTwo):
		weightsOne = genomeOne.network.params
		weightsTwo =  genomeTwo.network.params
		ratio = (genomeOne.fittness)/(genomeOne.fittness+genomeTwo.fittness)
		newWeights =  [ weightOne*ratio+(1-ratio)*weightsTwo[i] for i,weightOne in enumerate(weightsOne)]
		
		return genome(StateToActionNetwork(newWeights))

	def mutate(self):
	 	weights = self.network.params
		for i,w in enumerate(weights):
			if(random.uniform(0,1)<mutation_rate):
				weights[i] = w+evolutionMutate()
		self.network = StateToActionNetwork(weights)
	
 	
class helicopter_agent(Agent):
	randGenerator=Random()
	lastAction= None
	lastObservation= None
	rangeAction = None
	rangeObservation = None
	discountFactor = 1
        seed = None
	network = None
	#for record
	step = 0
	reward = 0
	episode_counter = 0
	chosenGenomeId= 0
       	#for evolution
	generation = []
        best = None		
	times = 0
        total_reward = 0
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
		 
		self.seed = StateToActionNetwork()
		self.best = self.seed
		self.generation = self.firstGeneration()	

	def agent_start(self,observation):
		#Generate random action, 0 or 1  
		global mutation_rate,variance
		self.lastAction = None
		self.lastObservation = None
		self.reward = 0.0
		self.episode_counter += 1.0
		self.step = 1.0
		self.times = 1
		mutation_rate =  mutation_rate*decay_rate
		
		self.network = self.getUnevaluatedGenome().network
		thisDoubleAction=self.agent_step_action(observation.doubleArray)
		
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
	 

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		 
		return returnAction
	
	def agent_step(self,reward, observation): 
		 		
		self.reward += reward
		self.step += 1
	        self.total_reward += reward
		
		thisDoubleAction=self.agent_step_action(observation.doubleArray)  
		if(self.isRisk(observation.doubleArray,thisDoubleAction)):
		 	self.times += 1
			thisDoubleAction = util.baselinePolicy(observation.doubleArray)  
			from pybrain.supervised.trainers import BackpropTrainer
			from pybrain.datasets import SupervisedDataSet
			ds = SupervisedDataSet(12, 4)
			ds.addSample(observation.doubleArray,self.best.activate(observation.doubleArray))	
			trainer = BackpropTrainer(self.network, ds)
			trainer.train()
  
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		 
		 
		self.lastObservation=copy.deepcopy(observation)
		self.lastAction=copy.deepcopy(returnAction)
		
		self.lastReward = reward 
		return returnAction
	
	def agent_step_action(self,observation):			
		return self.network.activate(observation)
	
	def isRisk(self,state,action):
		#if(state[0]>2 or state[0]<-2):
		#	return True
		#elif(state[1]>2 or state[1]<-2):
		#	return True
		#elif(state[2]>2 or state[2]<-2):
		#	return True
		#elif(state[3]>5 or state[3]<-5):
		#	return True
		#elif(state[4]>5 or state[4]<-5):
		#	return True
		#elif(state[5]>5 or state[5]<-5):
		#	return True
		#elif(state[6]>0.5 or state[6]<-0.5):
		#	return True
		#elif(state[7]>0.5 or state[7]<-0.5):
		#	return True
		#elif(state[8]>0.5 or state[8]<-0.5):
		#	return True
		#else:
		#	return False
		import math
		dis = math.sqrt(sum([pow(i,2) for i in state]))
		if(dis > 5):
			return True
		else:
			return False

	def firstGeneration(self):
		seed = self.seed.params
		generation = []
		for i in range(0,population_size):
			generation.append(genome(StateToActionNetwork([ w+evolutionMutate() for w in seed])))
		
		return generation

	def getUnevaluatedGenome(self):
		"""
			return random unevaluated genome

		"""
			 
		for i,g in enumerate(self.generation):
			if(g.fittness==0):
				self.chosenGenomeId = i
				return g
		
		

	def nextGeneration(self):
		nextGeneration = []
		generation_sorted = sorted(self.generation,key=lambda g:-g.fittness)
		for i in range(0,6):
			for j in range(i+1,6):
				genomeOne = generation_sorted[i]
				genomeTwo = generation_sorted[j]
				newGenome = genome.crossover(genomeOne,genomeTwo)
				newGenome.mutate()
				nextGeneration.append(newGenome)
		nextGeneration = numpy.append(nextGeneration,generation_sorted[0:5])
		return nextGeneration
	
	def getBestGenome(self):
		pairs = [ (i,g.fittness)for i,g in enumerate(self.generation)]
		total = sum([p[1] for p in pairs])
		randNumber = self.randGenerator.random()
		fittnessSofar = 0		
		for i,f in pairs:
			fittnessSofar += f/total
			if(fittnessSofar >= randNumber):
				return self.generation[i]

	
			
	def agent_end(self,reward):
		self.reward += reward 
		self.total_reward += reward
		self.generation[self.chosenGenomeId].fittness = 1.0/(0-self.reward/self.step)
		print self.reward,self.step,self.reward / self.step 
		print self.times
               # s = numpy.dot(self.seed.params[144:],self.network.params[144:])/numpy.linalg.norm(self.seed.params[144:])/numpy.linalg.norm(self.network.params[144:])
		 
                self.generation[self.chosenGenomeId].network = self.network
		for  g in self.generation:
			if(g.fittness == 0):
				return

		self.generation = self.nextGeneration()
		self.best = self.generation[15].network
		for i in self.generation[15:]:
			print i.fittness
			

	def agent_cleanup(self):
		print self.total_reward
	
	def agent_message(self,inMessage):
		pass
 	 
						

def evolutionMutate():
	return random.gauss(0,variance)	 
 
def StateToActionNetwork(genome=None):
	#initial a network [12,12,4] and initial weights are baseline policy versions
	
	from pybrain.structure import FeedForwardNetwork,LinearLayer,TanhLayer,FullConnection
	network = FeedForwardNetwork()
	inLayer= LinearLayer(12) 
	outLayer = LinearLayer(4)
	network.addInputModule(inLayer) 
	network.addOutputModule(outLayer)
	
	weights = [] 	
	if(genome == None):
		import pickle
		weights = pickle.load(open("seed2"))
	else:
		weights = genome
	 
	in_to_out = FullConnection(inLayer,outLayer)   		
	network.addConnection(in_to_out)
	network.sortModules()
	network._setParameters(weights)
	return network 		
 		

 

if __name__=="__main__":
	AgentLoader.loadAgent(helicopter_agent())
