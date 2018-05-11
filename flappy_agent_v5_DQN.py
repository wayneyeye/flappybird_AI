
from random import choice, random
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import time

## define the tf network here
tf.reset_default_graph()
n_inputs=4
n_hidden1=20
n_hidden2=40
n_hidden3=20
n_outputs=2
initializer=tf.contrib.layers.variance_scaling_initializer()
#the network
X=tf.placeholder(tf.float32,shape=[None,n_inputs])
hidden1=tf.layers.dense(X,n_hidden1,activation=tf.nn.elu,
					  kernel_initializer=initializer)
hidden2=tf.layers.dense(hidden1,n_hidden2,activation=tf.nn.elu,
					  kernel_initializer=initializer)
hidden3=tf.layers.dense(hidden2,n_hidden3,activation=tf.nn.elu,
					  kernel_initializer=initializer)
q_values=tf.layers.dense(hidden3,n_outputs,
					  kernel_initializer=initializer)
# which is estimated and recorded on the fly
q_target=tf.placeholder(tf.float32,shape=[None,n_outputs])

learning_rate=tf.placeholder(tf.float32,shape=[])
mse_loss=tf.reduce_mean(tf.squared_difference(q_values,q_target))
optimizer=tf.train.RMSPropOptimizer(learning_rate)

training_op=optimizer.minimize(mse_loss)
init=tf.global_variables_initializer()
saver=tf.train.Saver()

class flappy_agent():
	def __init__(self):
		'''initialize the agent and tf session'''
		# DQN hyper parameters
		self.iteration=0
		self.game_number=0
		self.n_iterations=3000 # after which the epsilon is forced to zero
		self.n_max_step=2500
		self.n_games_per_update=5
		self.save_per_iterations=100
		self.sample_interval=4
		self.discount_rate=0.95
		self.sess=tf.Session()
		self.epsilon=1
		self.epsilon_decay=0.01
		self.network_learning_rate=0.01
		self.min_network_learning_rate=0.000001
		self.network_decay=0.05
		self.flap_rate=0.2
		

		# DQN algorithm feeds
		self.reward_rate=0.1
		self.punishment=0
		self.all_actions=[] #
		self.all_rewards=[] #
		self.all_obs=[] #
		self.all_current_qsa=[]

		# DQN batch
		self.batch_size=50
		self.n_epochs=5


		# init network and logger
		self.sess.run(init)
		self.start_time=time.time()
		self.max_score=0
		self.max_score_log=[]
		self.last_100=[]
		self.last_100_avg_log=[]
		self.model_checkpoints_path="model_checkpoints/flappy_agent_dqn_05082018_v2_4_20_40_20_rep.ckpt"
		try:
			saver.restore(self.sess, self.model_checkpoints_path)
		except:
			print("no matched model checkpoints will start over")

	
	def new_round(self):
		'''initialize before each game'''
		self.game_number+=1


	def update_model(self):
		'''optimize the tf network every fixed intervals'''
		if self.game_number%self.n_games_per_update==0:
		# process in batch to speed up
			next_q_values_array=np.array(self.all_current_qsa[1:])
			# calculate q_target
			target_q_values_list=self.all_current_qsa.copy()
			# print(target_q_values_list)
			index=0
			# policy_learning_rate=max(0.05,self.policy_learning_rate/(1+self.iteration*self.policy_decay))
			for q,a,r in zip(self.all_current_qsa,self.all_actions,self.all_rewards):
				# if not survived
				if r<self.reward_rate:
					q_next_list=[0,0]
				else:
					q_next_list=list(next_q_values_array[index,:])
				q_new=r+self.discount_rate*max(q_next_list)
				# modify only the q(s,a) that has been executed
				target_q_values_list[index][a]=q_new
				index+=1
				
			# convert to numpy array
			target_q_values_array=np.array(target_q_values_list)
			obs_X=np.array(self.all_obs)
			# debug

			# if self.iteration==100:
			# 	print(self.all_rewards)
			# 	print(self.all_actions)
			# 	print(target_q_values_list)
			# 	print(q_next_list)

			current_lr=max(self.min_network_learning_rate,self.network_learning_rate/(1+self.iteration*self.network_decay))
			for epoch in range(self.n_epochs):
				obs_X,target_q_values_array=shuffle(obs_X,target_q_values_array)
				for i in range(obs_X.shape[0]//self.batch_size):
					obs_batch=obs_X[i*self.batch_size:i*self.batch_size+self.batch_size]
					target_q_batch=target_q_values_array[i*self.batch_size:i*self.batch_size+self.batch_size]
					feed_dict={X:obs_batch,q_target:target_q_batch,learning_rate:current_lr}
					self.sess.run(training_op,feed_dict=feed_dict)
			# reset all records after model update
			self.all_actions=[] #
			self.all_rewards=[] #
			self.all_obs=[] #
			self.all_current_qsa=[]
			# training iterations update
			self.iteration+=1
		else:
			pass

	def flap(self):
		'''when exploring '''
		if random()<self.flap_rate:
			return 1
		else:
			return 0
		
	def get_action(self,state):
		obs=self.stateWrapper(state)
		q_val=self.sess.run([q_values],
		feed_dict={X:obs.reshape(1,n_inputs)})
		#record current qsas
		self.all_current_qsa.append(q_val[0][0])
		# get the best action (epsilon greedy)
		if self.iteration<=self.n_iterations:
			if random()<self.epsilon/(1+self.iteration*self.epsilon_decay):
				best_action=self.flap()
			else:
				if q_val[0][0,1]>=q_val[0][0,0]:
					best_action=1
				else:
					best_action=0
		# get the best from q values
		else:
			if q_val[0][0,1]>=q_val[0][0,0]:
				best_action=1
			else:
				best_action=0
		# record the actions
		
		self.all_actions.append(best_action)
		return best_action

	def next_step(self,state):
		'''log rewards & states '''
		obs=self.stateWrapper(state)
		# survive=1 
		self.all_rewards.append(self.reward_rate)

		self.all_obs.append(obs)


	def end_round(self,score):
		# set the last reward to ...
		self.all_rewards[-1]=self.punishment   
	


	def debug(self):
		'''performance information send to stdout'''
		print("iteration {:3d} game {:5d} time_elapsed {:5.0f}s last_100_avg {:4.1f} max.score {:5d}".format\
			(self.iteration,self.game_number,self.elapsed_time,self.last_100_avg_log[-1],self.max_score),end='\r')

	def logger(self,score):
		'''log keeper to record game performance during training'''
		self.max_score=max(score,self.max_score)
		self.max_score_log.append(self.max_score)
		self.last_100.append(score)
		if len(self.last_100)>100:
			self.last_100=self.last_100[1:]
		self.last_100_avg_log.append(sum(self.last_100)/len(self.last_100))
		self.end_time=time.time()
		self.elapsed_time=self.end_time-self.start_time
		if self.iteration % self.save_per_iterations==0:
			saver.save(self.sess,self.model_checkpoints_path)
			pass


	def stateWrapper(self,state):
		'''simple division added to ensure the range of the input space'''
		return np.array([state['playery']/500,state['pipeY']/500,state['playerVelY']/10,state['pipeRightPos']/200])
	
	def get_state(self,state,reduce_factor=1):
		playerLeftPos=state['playerx']
		pipeRightPos=state['lowerPipes'][0]['x']+52
		pipeY=state['lowerPipes'][0]['y']
		if pipeRightPos<=playerLeftPos:
			pipeRightPos=state['lowerPipes'][1]['x']+52
			pipeY=state['lowerPipes'][1]['y']
		playery=state['playery']
		playerVelY=state['playerVelY']
		return {'player_pipe_Y':(playery-pipeY)//reduce_factor,
				'playery':playery//reduce_factor,
				'pipeY':pipeY//reduce_factor,
				'pipeRightPos':pipeRightPos//reduce_factor,
				'playerVelY':playerVelY,
				'crashed':state["crashed"],
				'score':state['score']}

agent=flappy_agent()