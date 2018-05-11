from random import choice, random
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import time
from collections import deque

tf.reset_default_graph()
## training parameters
n_hidden1=1000
n_hidden2=2000
n_hidden3=2000
n_hidden4=1000
n_outputs=2

## define the tf network here
def convpool(X,W,b):
	conv_out=tf.nn.conv2d(X,W,strides=[1,1,1,1],padding="SAME")
	conv_out=tf.nn.bias_add(conv_out,b)
	conv_out=tf.nn.elu(conv_out)
	pool_out=tf.nn.max_pool(conv_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	return pool_out

def init_filter(shape,poolsz):
	w=np.random.randn(*shape)/np.sqrt(np.prod(shape[:-1])+shape[-1]*np.prod(shape[:-2]/np.prod(poolsz)))
	return w.astype(np.float32)

from functools import partial
he_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG") # he init method
my_dense=partial(tf.contrib.layers.fully_connected,activation_fn=tf.nn.elu,
				weights_initializer=he_init)

cae_checkpoints_savepath="model_checkpoints_large/CAE_04302018_2layers_xsmall.ckpt"
checkpoints_savepath="model_checkpoints_large/DQN_cae_4hidden_05102018_v6.ckpt"
pool_sz=(2,2)

# cnn_pool layer 1
W1_shape=(4,4,3,10)
W1_init=init_filter(W1_shape,pool_sz)
b1_init=np.zeros(W1_shape[-1],dtype=np.float32)

# cnn_pool layer 2
W2_shape=(4,4,10,3)
W2_init=init_filter(W2_shape,pool_sz)
b2_init=np.zeros(W2_shape[-1],dtype=np.float32)
X=tf.placeholder(tf.float32,shape=(None,2,128,128,3),name="X")
traing_flag=tf.placeholder_with_default(False,shape=None,name="T_flag")

with tf.name_scope("cnn"):
	with tf.device("/gpu:0"):
		W1=tf.Variable(W1_init.astype(np.float32),trainable=False,name='W1')
		b1=tf.Variable(b1_init.astype(np.float32),trainable=False,name='b1')
		W2=tf.Variable(W2_init.astype(np.float32),trainable=False,name='W2')
		b2=tf.Variable(b2_init.astype(np.float32),trainable=False,name='b2')
	
	# first frame
	with tf.device("/gpu:0"):
		X1=tf.reshape(tf.slice(X,[0,0,0,0,0],[-1,1,-1,-1,-1]),[-1,128,128,3])
		Z11=convpool(X1,W1,b1)
		Z12=convpool(Z11,W2,b2)
	
	# second frame
	with tf.device("/gpu:0"):
		X2=tf.reshape(tf.slice(X,[0,1,0,0,0],[-1,1,-1,-1,-1]),[-1,128,128,3])
		Z21=convpool(X2,W1,b1)
		Z22=convpool(Z21,W2,b2)

with tf.name_scope("cnn_output"):
	with tf.device("/gpu:0"):
		# take the difference of two frames
		Z_diff=tf.contrib.layers.flatten(Z22-Z12)

with tf.name_scope("dense_layers"):
	with tf.device("/gpu:1"):
		# fully connected layer
		hidden1=my_dense(Z_diff,n_hidden1)
		hidden2=my_dense(hidden1,n_hidden2)
		hidden2_dropped=tf.layers.dropout(hidden2,rate=0.2,training=traing_flag)
		hidden3=my_dense(hidden2_dropped,n_hidden3)
		hidden4=my_dense(hidden3,n_hidden4)
		q_values=tf.contrib.layers.fully_connected(hidden4,n_outputs,
			activation_fn=None,weights_initializer=he_init)
		
with tf.name_scope("target_q"):
	with tf.device("/gpu:1"):
		q_target=tf.placeholder(tf.float32,shape=[None,n_outputs])
		
		
with tf.name_scope("training_op"):
	with tf.device("/gpu:1"):
		learning_rate=tf.placeholder(tf.float32,shape=[])
		mse_loss=tf.reduce_mean(tf.squared_difference(q_values,q_target))
		optimizer=tf.train.RMSPropOptimizer(learning_rate,momentum=0.9)
		training_op=optimizer.minimize(mse_loss)
		init=tf.global_variables_initializer()
		

with tf.name_scope("saver"):
	var_list={'cnn/Variable':W1,'cnn/Variable_1':b1,'cnn/Variable_2':W2,'cnn/Variable_3':b2}
	saver_cae_restore = tf.train.Saver(var_list=var_list)
	saver= tf.train.Saver()


class flappy_agent():
	def __init__(self):
		'''initialize the agent and tf session'''
		# DQN hyper parameters
		self.iteration=0
		self.game_number=0
		self.n_iterations=3000 # after which the epsilon is forced to zero
		self.n_max_step=1000
		self.n_games_per_update=5
		self.save_per_iterations=100
		self.sample_interval=8
		self.discount_rate=0.95
		self.sess=tf.Session()
		self.epsilon=1
		self.epsilon_decay=0.015
		self.network_learning_rate=0.0001
		self.min_network_learning_rate=0.000001
		self.network_decay=0.1
		self.flap_rate=0.45
		

		# DQN algorithm feeds
		self.reward_rate=0.1
		self.punishment=0
		self.all_actions=deque()#
		self.all_rewards=deque() #
		self.all_obs=deque() # <<- store 2-frame 
		self.all_current_qsa=deque()

		# DQN batch
		self.batch_size=25
		self.n_epochs=5


		# init network and logger
		self.sess.run(init)
		self.start_time=time.time()
		self.max_score=0
		self.max_score_log=[]
		self.last_100=deque()
		self.last_100_avg_log=[]
	
		try:
			saver_cae_restore.restore(self.sess, cae_checkpoints_savepath)
		except:
			print("restoring error, will start over!")

		try:
			saver.restore(self.sess, checkpoints_savepath)
		except:
			print("restoring error, will start over!")

	
	def new_round(self):
		'''initialize before each game'''
		self.game_number+=1


	def update_model(self):
		'''optimize the tf network every fixed intervals'''
		if self.game_number%self.n_games_per_update==0:
			# calculate q_target
			target_q_values_list=self.all_current_qsa.copy()
			# make next q array
			self.all_current_qsa.popleft()
			next_q_values_array=np.array(self.all_current_qsa)
			# print(target_q_values_list)
			index=0
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
				
			# convert to numpy array and cap at max steps to avoid OOM error
			# if len(target_q_values_list)>self.n_max_step:
			# 	target_q_values_list=target_q_values_list[-self.n_max_step:]
			# 	self.all_obs=self.all_obs[-self.n_max_step:]

			target_q_values_array=np.array(target_q_values_list)
			obs_X=np.array(self.all_obs)

			current_lr=max(self.min_network_learning_rate,self.network_learning_rate/(1+self.iteration*self.network_decay))
			for epoch in range(self.n_epochs):
				obs_X,target_q_values_array=shuffle(obs_X,target_q_values_array)
				for i in range(obs_X.shape[0]//self.batch_size):
					obs_batch=obs_X[i*self.batch_size:i*self.batch_size+self.batch_size]
					target_q_batch=target_q_values_array[i*self.batch_size:i*self.batch_size+self.batch_size]
					feed_dict={X:obs_batch,q_target:target_q_batch,learning_rate:current_lr}
					self.sess.run(training_op,feed_dict=feed_dict)
			
			# calculate loss
			self.loss=self.sess.run([mse_loss],
			feed_dict={X:obs_X,q_target:target_q_values_array,learning_rate:current_lr,traing_flag:True})[0]
			# reset all records after model update
			self.all_actions.clear()#
			self.all_rewards.clear() #
			self.obs_len=len(self.all_obs)
			self.all_obs.clear() #
			self.all_current_qsa.clear()
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
		
	def imitate_action(self,best_action,buffer_array):
		q_val=self.sess.run([q_values],
		feed_dict={X:buffer_array.reshape((1,)+buffer_array.shape)})
		#record current qsas
		self.all_current_qsa.append(q_val[0][0])
		if len(self.all_current_qsa)>self.n_max_step:
			self.all_current_qsa.popleft()
		# get the best action (epsilon greedy)
		self.all_actions.append(best_action)
		if len(self.all_actions)>self.n_max_step:
			self.all_actions.popleft()
		return best_action

	def get_action(self,state,buffer_array):
		q_val=self.sess.run([q_values],
		feed_dict={X:buffer_array.reshape((1,)+buffer_array.shape)})
		#record current qsas
		self.all_current_qsa.append(q_val[0][0])
		if len(self.all_current_qsa)>self.n_max_step:
			self.all_current_qsa.popleft()
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
		if len(self.all_actions)>self.n_max_step:
			self.all_actions.popleft()
		return best_action

	def next_step(self,state,buffer_array):
		'''log rewards & states '''
		obs=buffer_array
		# survive=1 
		self.all_rewards.append(self.reward_rate)
		if len(self.all_rewards)>self.n_max_step:
			self.all_rewards.popleft()
		self.all_obs.append(obs)
		if len(self.all_obs)>self.n_max_step:
			self.all_obs.popleft()

	def end_round(self,score):
		# set the last reward to ...
		self.all_rewards[-1]=self.punishment   
	

	def debug(self):
		'''performance information send to stdout'''
		if self.iteration>=1:
			print(""*100,end='\r')
			print("iter {:5d} game {:5d} elapsed {:5d}s last100_avg {:4.1f} max {:4d} loss {:6.2f} len_obs {:4d}".format\
				(self.iteration,self.game_number,int(self.elapsed_time),self.last_100_avg_log[-1],self.max_score,self.loss,self.obs_len),end='\r')

	def logger(self,score):
		'''log keeper to record game performance during training'''
		self.max_score=max(score,self.max_score)
		self.max_score_log.append(self.max_score)
		self.last_100.append(score)
		if len(self.last_100)>100:
			self.last_100.popleft()
		self.last_100_avg_log.append(sum(self.last_100)/len(self.last_100))
		self.end_time=time.time()
		self.elapsed_time=self.end_time-self.start_time
		if self.iteration % self.save_per_iterations==0:
			saver.save(self.sess,checkpoints_savepath)
			pass

	def get_state(self,state,reduce_factor=1):
		return {
				'crashed':state["crashed"],
				'score':state['score'],
				}

agent=flappy_agent()