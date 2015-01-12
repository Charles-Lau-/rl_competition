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
mutation_rate = 0.2

class genome():	
	def __init__(self,network,fittness=0):
		self.network = network
		self.fittness = fittness
	@staticmethod 
	def crossover(genomeOne,genomeTwo):
		weightsOne = genomeOne.network.params
		weightsTwo =  genomeTwo.network.params
		ratio = random.random()
		newWeights =  [ weightOne*ratio+(1-ratio)*weightsTwo[i] for i,weightOne in enumerate(weightsOne)]
		
		return genome(StateToActionNetwork(newWeights))

	def mutate(self):
	 	weights = self.network.params
		for i,w in enumerate(weights):
			if(random.uniform(0,1)>mutation_rate):
				weights[i] = w+util.evolutionMutate()
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
		self.generation = self.firstGeneration()	

	def agent_start(self,observation):
		#Generate random action, 0 or 1  
		self.lastAction = None
		self.lastObservation = None
		self.reward = 0.0
		self.episode_counter += 1.0
		self.step = 1.0
 
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
	       
                if(self.isRisk(observation.doubleArray)):
			thisDoubleAction = util.baselinePolicy(observation.doubleArray)
			
		else:
		 	thisDoubleAction=self.agent_step_action(observation.doubleArray)  
		  
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		 
		 
		self.lastObservation=copy.deepcopy(observation)
		self.lastAction=copy.deepcopy(returnAction)
		
		 
		return returnAction
	
	def agent_step_action(self,observation):			
		return self.network.activate(observation)
	
	def isRisk(self,state):
		for i in range(0,3):
			if(state[i]>3 or state[i]<-3):
				return True
		for i in range(3,6):
			if(state[i]>3 or state[i]<-3):
				return True
		for i in range(6,9):
			if(state[i]>3 or state[i]<-3):
				return True

		return False

	def firstGeneration(self):
		seed = self.seed.params
		generation = []
		for i in range(0,population_size):
			generation.append(genome(StateToActionNetwork([ w+util.evolutionMutate() for w in seed])))
		
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
		for i in range(0,population_size):
			genomeOne = self.getBestGenome()
			genomeTwo = self.getBestGenome()
			newGenome = genome.crossover(genomeOne,genomeTwo)
			newGenome.mutate()
			nextGeneration.append(newGenome)

		return nextGeneration
	
	def getBestGenome(self):
		pairs = [ (i,1.0/g.fittness)for i,g in enumerate(self.generation)]
		total = sum([p[1] for p in pairs])
		randNumber = self.randGenerator.random()
		fittnessSofar = 0		
		for i,f in pairs:
			fittnessSofar += f/total
			if(fittnessSofar >= randNumber):
				return self.generation[i]

	
			
	def agent_end(self,reward):
		self.reward += reward 
		self.generation[self.chosenGenomeId].fittness = (0-self.reward) / self.step
		print self.reward,self.step,self.reward / self.step
		if(self.step <5999):
			print self.lastObservation.doubleArray
			import time
			time.sleep(10)	

		for  g in self.generation:
			if(g.fittness == 0):
				return

		NextGeneration = self.nextGeneration()
		generation_sorted = sorted(self.generation,key=lambda g:g.fittness)
		for i in generation_sorted:
			print i.fittness
		for i in range(0,5):
			substitution_candidate = self.randGenerator.randint(0,population_size-1) 
			NextGeneration[substitution_candidate] = generation_sorted[i]
		self.generation = NextGeneration

	
	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		pass
 	 
						

	 
 
def StateToActionNetwork(genome=None):
	#initial a network [12,12,4] and initial weights are baseline policy versions
	
	from pybrain.structure import FeedForwardNetwork,LinearLayer,TanhLayer,FullConnection
	network = FeedForwardNetwork()
	inLayer= LinearLayer(12)
	hiddenLayer = LinearLayer(12)
	outLayer = TanhLayer(4)
	network.addInputModule(inLayer)
	network.addModule(hiddenLayer)
	network.addOutputModule(outLayer)
	
	weights = [] 	
	if(genome == None):
		import pickle
		weights = pickle.load(open("seed"))
	else:
		weights = genome
	 
	in_to_hidden = FullConnection(inLayer,hiddenLayer)   
	hidden_to_out = FullConnection(hiddenLayer,outLayer)
	for i in range(0,144):
		in_to_hidden.params[i] = weights[i]
	for j in range(0,48):
		hidden_to_out.params[j] = weights[j+144] 		
	network.addConnection(in_to_hidden)
	network.addConnection(hidden_to_out)
	network.sortModules()
	return network 		
 		

 

if __name__=="__main__":
	AgentLoader.loadAgent(helicopter_agent())
