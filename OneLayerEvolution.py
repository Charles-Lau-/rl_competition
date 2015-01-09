import random,numpy,util,sys,copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random


population_size = 20
mutation_rate = 0.3

class genome():	
	def __init__(self,network,fittness=0):
		self.network = network
		self.fittness = fittness
	@staticmethod 
	def crossover(genomeOne,genomeTwo):
		weightsOne = genomeOne.network.params
		weightsTwo =  genomeTwo.network.params
		newWeights =  [ weightOne*0.8+0.2*weightsTwo[i] for i,weightOne in enumerate(weightsOne)]
		
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
		
		print " " 
		thisDoubleAction=self.agent_step_action(observation.doubleArray)
		
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
	 

		self.lastAction=copy.deepcopy(returnAction)
		self.lastObservation=copy.deepcopy(observation)
		 
		return returnAction
	
	def agent_step(self,reward, observation): 
		 		
		self.reward += reward
		self.step += 1
		print reward
		 
		thisDoubleAction=self.agent_step_action(observation.doubleArray)  
		  
		returnAction=Action()
		returnAction.doubleArray = thisDoubleAction
		 
		 
		self.lastObservation=copy.deepcopy(observation)
		self.lastAction=copy.deepcopy(returnAction)
		
		 
		return returnAction
	
	def agent_step_action(self,observation):			
		return self.network.activate(observation)
	
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
		randNumber = self.randGenerator.randint(0,population_size-1)
		while(self.generation[randNumber].fittness !=0):
			randNumber = self.randGenerator.randint(0,population_size-1)
		
		self.chosenGenomeId = randNumber
		return self.generation[randNumber]

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
		self.generation[self.chosenGenomeId].fittness = self.reward / self.step
		print self.reward,self.step,self.reward / self.step
		
		if(self.episode_counter % 20 == 0):
			self.generation = self.nextGeneration()
	

	def agent_cleanup(self):
		pass
	
	def agent_message(self,inMessage):
		pass
 	 
						

	 
 
def StateToActionNetwork(genome=None):
	#initial a network [12,12,4] and initial weights are baseline policy versions
	
	from pybrain.structure import FeedForwardNetwork,LinearLayer,TanhLayer,FullConnection
	network = FeedForwardNetwork()
	inLayer= LinearLayer(12)
	hiddenLayer = TanhLayer(12)
	outLayer = TanhLayer(4)
	network.addInputModule(inLayer)
	network.addModule(hiddenLayer)
	network.addOutputModule(outLayer)
	#inital weights
	initial_weights ="-8.67774780e-01  -1.06126675e+00  -1.15673246e-01  2.32831289e-01 \
  			-1.33417722e-01   1.35235438e-01   4.52656330e-01  -7.18928247e-01 \
 	 		-1.42420455e+00  -4.47104061e-01   5.73706437e-01  -7.36605714e-01 \
  			-7.34918094e-01  -3.88001465e-01  -2.71370466e-01  -1.39807104e+00 \
  			 4.69457821e-01  -1.80894004e+00   5.32996686e-01   7.42656505e-01 \
 			 -6.85283868e-02  -1.20336101e+00   4.67082614e-01  -1.20566640e-01 \
  			 7.99478484e-01  -1.18125203e-01   7.47896020e-02   1.07364538e+00 \
 			 -2.96465060e-01  -1.88933997e+00   1.42265931e+00  -2.50172397e-01 \
  			-3.78646157e-01   6.26063491e-01  -1.88568246e+00  -2.68558950e-01 \
  			-1.56838488e+00  -5.96599224e-02   2.64931540e-01   7.35243520e-01 \
  			-4.36337124e-02  -6.06494467e-01  -2.82567916e+00   4.65097564e-01 \
  			 1.30868079e+00  -2.87431872e+00  -1.97732740e-01  -9.85536503e-01 \
  			 8.69147209e-02  -1.21702375e-01   4.00394306e-02   1.74995347e+00 \
  			-6.43077546e-01  -5.84016435e-01   5.68659344e-01  -1.82790828e+00 \
  			-1.01562839e-01  -5.62585486e-01  -1.03496016e+00   1.48453391e+00 \
  			 1.50266315e+00  -1.55671505e+00   1.21984299e+00  -1.71678390e+00 \
  			 6.13576249e-01  -2.47223780e-01   8.97057068e-01   8.01841419e-01 \
  			-5.01024114e-02   7.34958727e-01  -1.79572718e-01   3.91851983e-01 \
  -			8.38448061e-01  -3.92438857e-01  -4.10666969e-01  -1.50514442e+00 \
  			-4.29149367e-01  -6.63605126e-01  -6.48818888e-01  -4.70973329e-02 \
   			1.61334205e+00  -3.19072746e-01  -1.59413422e+00  -2.03534035e+00 \
   			1.12068548e+00   1.19860743e+00  -9.26304382e-02   2.66652326e-01 \
  			-2.27000650e-01  -7.44797141e-01   3.94927174e-01   1.37642596e+00 \
   			 2.82337738e-01  -6.20660644e-01   1.45771455e+00  -8.70830638e-03 \
  			-1.12215411e+00   7.20217537e-01   1.09573009e-01  -1.64029476e-01 \
  			-5.17286486e-02   1.80363426e+00   8.89720736e-01  -1.36262611e+00 \
   			 1.59299856e-01  -8.99861020e-01  -1.60404287e-01   1.60774005e-01 \
   			-1.62177067e-01   4.93455690e-01   2.74683582e+00   3.07558215e-02 \
  			-8.14037184e-01  -5.74580455e-01   7.50756944e-01  -3.98436277e-01 \
  			-1.05195388e-01  -1.10257008e+00   3.88067288e-02  -7.46809294e-01 \
  -			3.20848385e-01  -2.71567743e+00  -1.86178460e+00   1.34636616e+00 \
   			4.64757600e-01   1.12179494e+00   7.92946468e-01  -3.46556097e-01 \
   			7.36096661e-01  -8.37969849e-01  -9.16219904e-01  -3.32538886e-02 \
   			1.34310170e+00  -1.48769932e+00   1.19408268e-01   1.68983258e-01 \
  			 2.91484582e-02  -6.04045841e-01   6.37242792e-01  -1.02148785e+00 \
  			-1.43926986e+00   5.20183148e-01  -1.01742541e+00  -3.39322979e-01 \
  			-4.14840208e-02   4.68579374e-02  -7.03792519e-02   5.63431103e-02 \
   			4.85567026e-02   1.25852674e-02   1.64220129e-02  -2.53130126e-02 \
   			2.42077327e-02  -9.37077940e-03  -4.84253135e-02   7.64875847e-02 \
   			7.57643791e-02  -1.49685614e-02  -3.38807998e-02   2.85944706e-02 \
  			 1.68855602e-02   5.21616338e-02  -1.31630794e-03   5.28125829e-02 \
   			5.36709803e-03   3.08505773e-02   4.14951096e-02  -5.34971972e-02 \
  			 1.44647161e-01  -4.23931031e-02   4.04126278e-02  -7.10746724e-02 \
  			-5.41653277e-02  -2.44358941e-02  -7.51407175e-02   7.50540102e-03 \
  			-3.93433761e-02  -3.62153348e-02  -8.49413062e-02  -8.63981390e-03 \
   			4.73963775e-02  -6.42754751e-02   2.00311914e-03   2.14777057e-02 \
  			-1.91331058e-02   1.13850137e-01  -1.88856956e-02  -1.84348783e-03 \
   			6.48878407e-02   3.17626887e-01   2.12471711e-02  -2.70431690e-02"
  
	weights = []
	#if no specific weights , then we initialize weights as baseline policy
	#otherwise we set it as same as genome
	if(genome==None):
		for i in initial_weights.split("  "):
			num = i.strip().replace('\t','')
			try:
				weights.append(float(num))
			except:
				pass 
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
