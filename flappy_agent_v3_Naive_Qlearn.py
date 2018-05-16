import time
from random import choice, random
from collections import deque
import pickle

class flappy_agent():
	def __init__(self,sample_interval=8):
		# agent sample interval (frames)
		self.iteration=0
		self.game_number=0
		self.n_iterations=300 # after which the epsilon is forced to zero
		self.n_max_step=100 # length of memory
		self.sample_interval=sample_interval
		self.save_per_iterations=10

		# model update parameters
		self.discount_rate=0.9 #gamma
		self.learning_rate=0.4 #alpha
		self.learning_rate_min=0.01
		self.learning_rate_decay=3
		self.current_lr=self.learning_rate
		self.epsilon=1
		self.epsilon_decay=3
		self.current_epsilon=self.epsilon
		self.flap_rate=0.45 # random exploration

		# q-table
		self.Q_table={}
		try:
			self.load_dict()
		except:
			print("no matched pickle file will start over!")

		# sars recording
		self.reward_list=deque()
		self.action_list=deque()
		self.state_list=deque()

		# performance
		self.start_time=time.time()
		self.elapsed_time=0
		self.max_score=0
		self.last_100=[]
		self.max_difference=-1
		self.sum_difference=0
		self.max_diff_calculate_flag=False

		self.max_score_log=[]
		self.iter_log=[]
		self.game_number_log=[]
		self.last_100_avg_log=[]
		self.last_10_avg_log=[]
		self.last_1_avg_log=[]
		self.elapsed_time_log=[]
		self.max_diff_log=[]
		self.sum_diff_log=[]
		self.current_lr_log=[]
		self.current_epsilon_log=[]

		# punishiment if crash
		self.reward_rate=1
		self.punishment=-1
		
	def save_dict(self):
		pickle.dump( agent.Q_table, open("model_checkpoints/"+agent.model_prefix+agent.model_suffix+".pickle", "wb"))

	def load_dict(self):
		self.Q_table=pickle.load(open("model_checkpoints/"+agent.model_prefix+agent.model_suffix+".pickle", "rb"))

	def new_round(self):
		'''initialize'''
		self.game_number+=1

	def end_round(self):
		self.reward_list[-1]=self.punishment
	
	def flap(self):
		if random()<self.flap_rate:
			return 1
		else:
			return 0
		
	def update_model(self):
		if len(self.state_list)==self.n_max_step:
			# initialize unknown state-reward values and epsilon values
			self.max_difference=0
			self.sum_difference=0
			for s in self.state_list:
				state_code=self.stateEncoder(s)
				if state_code not in self.Q_table:
					self.Q_table[state_code]={}
					self.Q_table[state_code][0]=0.0
					self.Q_table[state_code][1]=0.0
					self.Q_table[state_code]['ct']=0
					self.Q_table[state_code]['current_lr']=self.learning_rate
					self.Q_table[state_code]['current_epsilon']=self.epsilon
			# update Q_table
			current_iter=0
			for s,a,r in zip(self.state_list,self.action_list,self.reward_list):
				state_code=self.stateEncoder(s)
				if r<self.reward_rate: # crashed state
					max_Qsa_value=0
				else:
					next_state_code=self.stateEncoder(self.state_list[current_iter+1])
					max_Qsa_value=max(self.Q_table[next_state_code][0],self.Q_table[next_state_code][1])       
				
				# update Q(s,a)
				# self.current_lr=max(self.learning_rate_min,self.learning_rate/(1+self.Q_table[state_code]['ct']*self.learning_rate_decay))
				# update learning rate and epsilon
				self.Q_table[state_code]['current_lr']=self.Q_table[state_code]['current_lr']-max(0,self.Q_table[state_code]['current_lr']-\
					self.learning_rate_min)/max(1,self.n_iterations-self.Q_table[state_code]['ct'])*self.learning_rate_decay
				self.current_lr=self.Q_table[state_code]['current_lr']
				self.Q_table[state_code]['current_epsilon']=self.Q_table[state_code]['current_epsilon']-(self.Q_table[state_code]['current_epsilon'])/max(1,self.n_iterations-self.Q_table[state_code]['ct'])*self.epsilon_decay
				self.old_Qsa=self.Q_table[state_code][a]
				self.Q_table[state_code][a]=(1-self.current_lr)*self.Q_table[state_code][a]+self.current_lr*(r+self.discount_rate*max_Qsa_value)
				self.Q_table[state_code]['ct']+=1 # update count for learning rate and epsilon decay
				self.max_difference=max(self.max_difference,abs(self.Q_table[state_code][a]-self.old_Qsa))
				self.sum_difference+=abs(self.Q_table[state_code][a]-self.old_Qsa)
				current_iter+=1

			self.iteration+=1
			self.state_list.clear()
			self.action_list.clear()
			self.reward_list.clear()
			# sign to update loss log
			self.max_diff_calculate_flag=True
		else:
			pass


	def get_action(self,state):
		state_code=self.stateEncoder(state)
		if state_code in self.Q_table:
			if random()<=self.Q_table[state_code]['current_epsilon']:
				best_action=self.flap()
			else:
				if self.Q_table[state_code][1]>self.Q_table[state_code][0]:
					best_action=1
				else:
					best_action=0
		else:
			best_action=self.flap()
		self.action_list.append(best_action)
		if len(self.action_list)>self.n_max_step:
			self.action_list.popleft()
		return best_action


	def next_step(self,state):
		self.reward_list.append(self.reward_rate)
		if len(self.reward_list)>self.n_max_step:
			self.reward_list.popleft()
		self.state_list.append(state)
		if len(self.state_list)>self.n_max_step:
			self.state_list.popleft()
	
		

	def debug(self):
		print("iteration {:3d} game {:5d} time_elapsed {:5.0f}s last_100_avg {:4.1f} max.score {:5d}".format\
			(self.iteration,self.game_number,self.elapsed_time,self.last_100_avg_log[-1],self.max_score),end='\r')

	def stateEncoder(self,state):
		return str(state['player_pipe_Y'])+'#'+str(state['pipeRightPos'])+'#'+str(state['playerVelY'])
	
	def get_state(self,state,reduce_factor=16):
		playerLeftPos=state['playerx']
		pipeRightPos=state['lowerPipes'][0]['x']+52
		pipeY=state['lowerPipes'][0]['y']
		if pipeRightPos<=playerLeftPos:
			pipeRightPos=state['lowerPipes'][1]['x']+52
			pipeY=state['lowerPipes'][1]['y']
		playery=state['playery']
		playerVelY=state['playerVelY']
		self.last_score=state["score"]
		return {'player_pipe_Y':int((playery-pipeY)//20), # negative means the bird is higher than the pipe
				'pipeRightPos':int(pipeRightPos//24),
				'playerVelY':int(playerVelY//3),
				'playery':int(playery//40),
				'pipeY':int(pipeY//40),
				'crashed':state["crashed"]}

	def logger(self,score):
		self.max_score=max(score,self.max_score)
		self.last_100.append(score)
		if len(self.last_100)>100:
			self.last_100=self.last_100[1:]
		self.end_time=time.time()
		self.elapsed_time=self.elapsed_time+self.end_time-self.start_time
		self.start_time=self.end_time
		if self.iteration % self.save_per_iterations==0: ### Mod needed
			self.save_dict()
			pass
		# append to log caching
		self.iter_log.append(self.iteration)
		self.game_number_log.append(self.game_number)
		self.elapsed_time_log.append(self.elapsed_time)
		self.last_100_avg_log.append(sum(self.last_100[-min(100,len(self.last_100)):])/min(100,len(self.last_100)))
		self.last_10_avg_log.append(sum(self.last_100[-min(10,len(self.last_100)):])/min(10,len(self.last_100)))
		self.last_1_avg_log.append(sum(self.last_100[-min(1,len(self.last_100)):])/min(1,len(self.last_100)))
		self.max_score_log.append(self.max_score)
		self.max_diff_log.append(self.max_difference)
		self.sum_diff_log.append(self.sum_difference)
		
agent=flappy_agent()