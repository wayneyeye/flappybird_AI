
from random import choice, random
import tensorflow as tf
import numpy as np

## define the tf network here
tf.reset_default_graph()
n_inputs=4
n_hidden1=10
n_hidden2=10
n_hidden3=5
n_outputs=2
initializer=tf.contrib.layers.variance_scaling_initializer()
#the network
X=tf.placeholder(tf.float32,shape=[None,n_inputs])
hidden1=tf.layers.dense(X,n_hidden1,activation=tf.nn.elu,
					  kernel_initializer=initializer)
hidden2=tf.layers.dense(hidden1,n_hidden2,activation=tf.nn.elu,
					  kernel_initializer=initializer)
q_values=tf.layers.dense(hidden2,n_outputs,
					  kernel_initializer=initializer)
# which is estimated and recorded on the fly
q_target=tf.placeholder(tf.float32,shape=[None,n_outputs])

learning_rate=tf.placeholder(tf.float32,shape=[])
mse_loss=tf.reduce_mean(tf.squared_difference(q_values,q_target))
optimizer=tf.train.AdamOptimizer(learning_rate)

training_op=optimizer.minimize(mse_loss)
init=tf.global_variables_initializer()
saver=tf.train.Saver()

class flappy_agent():
	def __init__(self):
		'''initialize the agent and tf session'''
		# DQN hyper parameters
		self.iteration=0
		self.game_number=0
		self.n_iterations=2000 # after which the epsilon is forced to zero
		self.n_max_step=10000
		self.n_games_per_update=5
		self.save_per_iterations=100
		self.sample_interval=4
		self.network_learning_rate=0.3
		self.discount_rate=0.97
		self.policy_learning_rate=0.5
		self.sess=tf.Session()
		self.epsilon=1
		self.epsilon_decay=0.01
		self.network_decay=0.01
		self.policy_decay=0.01
		self.flap_rate=0.2

		# DQN algorithm feeds
		self.all_actions=[] #
		self.reward_rate=0.1
		self.all_rewards=[] #
		self.all_obs=[] #
		self.all_current_qsa=[]

		self.sess.run(init)
		self.max_score=0
		self.max_score_log=[]
		self.last_100=[]
		self.last_100_avg_log=[]
		self.model_checkpoints_path="model_checkpoints/flappy_agent_dqn_05072018_v1.ckpt"
		try:
			saver.restore(self.sess, self.model_checkpoints_path)
		except:
			print("no matched model checkpoints will start over")

	
	def new_round(self):
		'''initialize before each game'''
		self.game_number+=1
		

	def update_model(self):
		'''optimize the tf network every fixed intervals'''
		# covenrt obs array
		next_obs_array=np.array(self.all_obs[1:])
		# process in batch to speed up
		next_q_values_array=self.sess.run([q_values],feed_dict={X:next_obs_array})[0]
		# calculate q_target
		target_q_values_list=self.all_current_qsa.copy()
		# print(target_q_values_list)
		index=0
		policy_learning_rate=max(0.05,self.policy_learning_rate/(1+self.iteration*self.policy_decay))
		for q,a,r in zip(self.all_current_qsa,self.all_actions,self.all_rewards):
			q_next_list=list(next_q_values_array[index,:])
			# if not survived
			if r<self.reward_rate:
				q_next_list=[0,0]
			q_new=(1-policy_learning_rate)*q[a]+policy_learning_rate*(r+self.discount_rate*max(q_next_list))
			# modify only the q(s,a) that has been executed
			target_q_values_list[index][a]=q_new
			index+=1
			if index==next_q_values_array.shape[0]-1:
				target_q_values_list[index][a]=target_q_values_list[index][a]*(1-policy_learning_rate)+policy_learning_rate*r
				break
		# convert to numpy array
		target_q_values_array=np.array(target_q_values_list)

		
		# DQN Training ops
		if self.game_number%self.n_games_per_update==0:
			if self.iteration==5:
				print(self.all_rewards)
				print(self.all_actions)
				print(target_q_values_list)
				print(q_next_list)

			current_lr=max(0.05,self.network_learning_rate/(1+self.iteration*self.network_decay))
			feed_dict={X:np.array(self.all_obs),q_target:target_q_values_array,learning_rate:current_lr}
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
		# record the estimated q_val
		
		self.all_actions.append(best_action)
		return best_action

	def next_step(self,state):
		'''log rewards & states '''
		obs=self.stateWrapper(state)
		# survive=1 
		self.all_rewards.append(self.reward_rate)
		self.all_obs.append(obs)

	def end_round(self):
		# set the last reward to ...
		self.all_rewards[-1]=-3    

	def debug(self):
		'''performance information send to stdout'''
		print("iteration {:3d} game {:5d} last_100_avg {:4.1f} max.score {:5d}".format\
			(self.iteration,self.game_number,self.last_100_avg_log[-1],self.max_score),end='\r')

	def logger(self,score):
		'''log keeper to record game performance during training'''
		self.max_score=max(score,self.max_score)
		self.max_score_log.append(self.max_score)
		self.last_100.append(score)
		if len(self.last_100)>100:
			self.last_100=self.last_100[1:]
		self.last_100_avg_log.append(sum(self.last_100)/len(self.last_100))
		if self.iteration % self.save_per_iterations==0:
			# saver.save(self.sess,self.model_checkpoints_path)


	def stateWrapper(self,state):
		'''simple division added to ensure the range of the input space'''
		return np.array([state['playery']/1000,state['pipeY']/1000,state['playerVelY']/20,state['pipeRightPos']/512])
	
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