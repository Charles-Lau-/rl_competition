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
	#inital weights
	initial_weights =    """5.77837680e-02   7.06018185e-01  -1.46700853e-01  -9.65923829e-02  
				9.61787572e-03   2.65688219e-01   4.61255221e-01  -9.14878082e-01
   				1.24124995e+00  -1.65912884e+00  -9.85109303e-01  -7.80173060e-01
  				-1.59624147e+00   6.76936709e-01   4.86168957e-02   4.99892112e-02
  				-5.46417818e-02   5.68308169e-01   7.33885087e-01   1.49175977e+00
 				 -1.02034745e+00   4.16901913e-01   1.31154568e+00   1.03563982e+00
  				-1.63436043e-01  -1.00101511e+00  -9.63648823e-01  -2.22176807e-01
   				8.04624759e-02  -4.61370951e-01  -7.86777612e-01  -1.22148594e-01
  				-3.31872556e-01  -8.55333398e-02   2.99508703e+00  -8.06389487e-01
   				7.04265005e-01  -1.22301872e+00   1.90570809e-01   4.54918324e-01
  				-4.17337466e-02  -6.33409448e-01   9.67563392e-01  -7.30323211e-01
  				-1.40684878e+00  -6.97437344e-01  -5.42994459e-01   3.58375684e-01
   				1.86921576e-01   1.67013656e-01  -2.28165316e-01   3.00210162e-02
  				4.64725506e-02   5.45300597e-03  -2.49279103e-01   1.77965608e+00
   				1.01192599e+00  -2.11313309e-01  -1.68621756e+00  -6.22862351e-01
   				1.50482726e-01   1.27020926e-02   9.86828798e-01   3.99364028e-01
  				-1.55356849e-01   1.11038382e-02  -7.41806124e-01  -6.17463233e-01
  				-2.65096450e-01  -1.20224960e+00   1.39066140e+00  -5.14017486e-01
  				-6.38745936e-01  -1.95035315e-01   4.84585361e-01   1.87805344e-01
  				-1.21682288e-01   1.21885105e-01   3.50327405e-02   2.38393565e-01
   				4.95001028e-01   1.70929033e+00  -3.91373726e-01   1.14371800e+00
   				1.53314058e+00   1.20219806e-01   2.95939802e-01  -2.83335682e-01
   				8.02747833e-02  -8.59657681e-02  -5.42110285e-01   6.71557129e-01
  				-6.31636845e-01   9.65407739e-01   9.00393992e-01   2.06055430e-01
   				7.46671118e-02  -7.18494710e-01   1.18133060e+00   4.15187944e-01
  				-1.79438810e-01  -1.71385631e-01   3.34426295e-01  -2.00574170e-02
  				-8.51039262e-01   1.75330963e+00  -3.36063691e+00   3.46925104e-01
  				 1.67540271e+00   1.03899510e-01  -1.54018530e-01  -5.06443917e-01
   				1.41048410e-01  -2.36816984e-01   7.16675277e-01  -6.09312953e-01
   				1.47215316e+00   3.92558038e-01   5.40620147e-01  -1.09742816e+00
 				-8.15658418e-02  -1.22629845e+00  -8.63395842e-01  -1.88595221e-02
   				8.93978508e-02  -6.44015879e-01   5.53754829e-01  -1.20731279e+00
   				6.45644709e-01   1.34695414e+00  -5.75938099e-01  -1.55505241e+00
  				-8.20698558e-01   9.23985362e-01   1.57729826e+00   3.16217741e-01
  				-2.65920603e-01   6.50787223e-01   1.00313869e+00   4.41196066e-02
   				3.19658565e-01  -4.95562041e-01  -1.66290456e+00   6.80170176e-01
   				1.30154409e-01  -1.17718679e-01   3.11766455e-01   1.08064554e-01
   				5.57824561e-02  -1.35221817e-01  -3.26102442e-02  -6.66646068e-02
   				4.19394688e-02  -7.51532566e-03  -2.22807914e-01   1.64123208e-01
  				-8.69039518e-02   8.24719809e-02   3.83865485e-02   1.24265896e-02
  				-2.79075967e-02   1.07774402e-01   5.29672791e-02  -3.03904891e-02
  				-8.51628555e-02   1.07795754e-01  -2.76723846e-04   3.63062562e-03
  				1.12598894e-01   8.86916074e-02   2.84222259e-02  -2.54768122e-02
   				2.48618165e-02   7.19669217e-02  -2.92676549e-02   5.03289768e-02
   				7.73790093e-02   6.40381136e-03   5.33435251e-02  -4.19214926e-02
  				-9.08925142e-02   1.86064739e-02  -6.67785040e-02  -1.03374457e-01
  				-9.62126496e-02  -1.23730143e-02   1.30286649e-01   9.97029354e-02
   				1.27250023e-01   8.00670327e-02   4.63841292e-02   5.53386785e-02"""
  
	weights = [] 	
	if(genome == None):
		for i in initial_weights.split(" "):
			num = i.strip().replace('\t','')
			num = num.replace('\n','')
			num = num.strip()
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
