import random,numpy,util,sys,copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random


population_size = 20
mutation_rate = 0.4

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
	initial_weights = """-0.82254189 -1.13179445 -0.05073786  0.26591425 -0.1284119   0.08943027
 			      0.42276271 -0.7071644  -1.45276617 -0.44496227  0.59200697 -0.76490859
 			     -0.82167338 -0.39902595 -0.34932747 -1.5301132   0.4874284  -1.75511689
  			      0.48486169  0.81363237 -0.15560306 -1.28014402  0.432026   -0.245171
  			      0.78031838 -0.15817382  0.11117014  1.04968207 -0.2928946  -1.83646693
  			      1.44371163 -0.16511239 -0.28240856  0.66388542 -1.82       -0.31922762
 			     -1.62204838 -0.12312791  0.22280484  0.7411014  -0.05863777 -0.5328774
 			     -2.78973853  0.46491466  1.42202806 -2.93244732 -0.24784862 -1.09437463
 			     -0.07650338 -0.22306763 -0.0183736   1.78186929 -0.67513165 -0.55366751
 			      0.5925835  -1.82412031 -0.05014905 -0.53091446 -1.00910792  1.35670824
 			      1.58071742 -1.65940386  1.17890985 -1.64222732  0.56357161 -0.304062
 			      0.83628703  0.80113178 -0.09179465  0.74009935 -0.34146348  0.45395903
 			     -0.80083394 -0.3178608  -0.46622104 -1.59866551 -0.3893681  -0.67853351
 			     -0.61304091 -0.12200701  1.65532154 -0.33992727 -1.56087088 -2.02031568
  			      1.02997029  1.21253299 -0.06733012  0.28724485 -0.27014336 -0.83057191
 			      0.39323538  1.30558669  0.26726448 -0.65961121  1.54584633  0.09210854
			     -0.99081429  0.59634696  0.08429763 -0.1085911   0.01785386  1.78681155
 			      0.87636657 -1.2413246   0.08531575 -0.92648945 -0.20240477  0.16277405
 			     -0.23555818  0.49663751  2.75629226  0.03482599 -0.64754342 -0.61886618
  			      0.77400906 -0.28588506 -0.18226857 -1.15349435  0.09339758 -0.84021149
 			     -0.3769615  -2.71952741 -1.92806955  1.33770349  0.46549708  1.0234573
  			      0.77816064 -0.36149316  0.65660944 -0.79934234 -0.85783489 -0.10840895
 			      1.35537789 -1.50803792  0.10239295  0.20335467  0.07891178 -0.56889871
  			      0.59446914 -1.07917626 -1.44565869  0.46396979 -1.0022648  -0.36037274
  			      0.04604105  0.10828613 -0.09156346  0.05961271  0.07350161  0.03483664
  			      0.01918546  0.029282   -0.03780433  0.01140018 -0.04217829  0.12422228
  			      0.10494436  0.03090324  0.02751887  0.1757922  -0.1175768   0.04984245
                              0.03805592  0.07699565  0.0927753   0.03017363 -0.02785207 -0.08504634
 		              0.23548627 -0.024849    0.08893206 -0.02284833  0.04222917 -0.01530065
 			     -0.0336135   0.08849411 -0.02291273 -0.05779803 -0.01868145  0.00836078
 			      0.08720535 -0.11581814 -0.03772317  0.05162675  0.10993543  0.08677515
     			     -0.03086664 -0.02367544  0.10032227  0.15426584 -0.03793561 -0.07125042"""
  
	weights = [] 
	for i in initial_weights.split(" "):
		num = i.strip().replace('\t','')
		num = num.replace('\n',' ')
		num = num.strip()
		try:
			weights.append(float(num))
		except:
			pass 
 
	 
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
