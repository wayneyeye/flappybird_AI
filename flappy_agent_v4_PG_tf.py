
from random import choice, random
import tensorflow as tf
import numpy as np

## define the tf network here
tf.reset_default_graph()
n_inputs=4
n_hidden1=10
n_hidden2=5
n_outputs=1
initializer=tf.contrib.layers.variance_scaling_initializer()
#the network
X=tf.placeholder(tf.float32,shape=[None,n_inputs])
hidden1=tf.layers.dense(X,n_hidden1,activation=tf.nn.elu,
                      kernel_initializer=initializer)
hidden2=tf.layers.dense(hidden1,n_hidden2,activation=tf.nn.elu,
                      kernel_initializer=initializer)
logits=tf.layers.dense(hidden2,n_outputs,
                      kernel_initializer=initializer)
outputs=tf.nn.sigmoid(logits)
p_flap_not=tf.concat(axis=1,values=[outputs,1-outputs])
action=tf.multinomial(tf.log(p_flap_not),num_samples=1)

#take the random action as the best action

y=1.0-tf.to_float(action)
learning_rate=0.05
cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                      logits=logits)
optimizer=tf.train.AdamOptimizer(learning_rate)
grads_and_vars=optimizer.compute_gradients(cross_entropy)

gradients=[grad for grad,variable in grads_and_vars]

gradient_placeholders=[]
grads_and_vars_feed=[]
for grad,variable in grads_and_vars:
    gradient_placeholder=tf.placeholder(tf.float32,shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder,variable))

training_op=optimizer.apply_gradients(grads_and_vars_feed)
init=tf.global_variables_initializer()
saver=tf.train.Saver()


class flappy_agent():
    def __init__(self):
        '''initialize the agent and tf session'''
        self.iteration=0
        self.game_number=0
        self.n_iterations=250
        self.n_max_step=1000
        self.n_games_per_update=5
        self.save_per_iterations=100
        self.sample_interval=4
        self.discount_rate=0.95
        self.sess=tf.Session()
        self.all_rewards=[]
        self.all_gradients=[]
        self.sess.run(init)
        self.max_score=0
        self.max_score_log=[]
        self.last_100=[]
        self.last_100_avg_log=[]
        self.model_checkpoints_path="model_checkpoints/flappy_agent_pg_s4_4105_042718_5.ckpt"
        try:
            saver.restore(self.sess, self.model_checkpoints_path)
        except:
            print("no matched model checkpoints will start over")


    def discount_rewards(self,rewards,discount_rate):
        '''calculate the discounted rewards'''
        discounted_rewards=np.empty(len(rewards))
        cumulative_rewards=-3
        for step in reversed(range(len(rewards))):
            cumulative_rewards=rewards[step]+cumulative_rewards*discount_rate
            discounted_rewards[step]=cumulative_rewards
        return discounted_rewards

    def discount_and_normalize_rewards(self,all_rewards,discount_rate):
        '''return normalized discounted rewards'''
        all_discounted_rewards=[self.discount_rewards(rewards,discount_rate)
                               for rewards in all_rewards]
        flat_rewards=np.concatenate(all_discounted_rewards)
        reward_mean=flat_rewards.mean()
        reward_std=flat_rewards.std()
        return [(discount_rewards-reward_mean)/reward_std
                for discount_rewards in all_discounted_rewards]
    
    def new_round(self):
        '''initialize before each game'''
        self.game_number+=1
        self.current_rewards=[]
        self.current_gradients=[]

        
    def policy_gradient(self):
        '''optimize the tf network every fixed intervals'''
        if self.game_number%self.n_games_per_update==0:
            all_rewards=self.discount_and_normalize_rewards(self.all_rewards,self.discount_rate)
            feed_dict={}
            for var_index,grad_placeholder in enumerate(gradient_placeholders):
                mean_gradients=np.mean(
                [reward*self.all_gradients[game_index][step][var_index]
                for game_index,rewards in enumerate(all_rewards) 
                for step,reward in enumerate(rewards)],axis=0
                )
                feed_dict[grad_placeholder]=mean_gradients
            self.sess.run(training_op,feed_dict=feed_dict)
            # reset all rewards
            # reset all gradients
            self.all_rewards=[]
            self.all_gradients=[]
            self.iteration+=1
        else:
            pass

        
    def get_action(self,state):
        obs=self.stateWrapper(state)
        action_val,gradients_val=self.sess.run([action,gradients],
        feed_dict={X:obs.reshape(1,n_inputs)})
        self.current_gradients.append(gradients_val)
        return action_val[0][0]

    def next_step(self):
        '''log rewards when policy'''
        self.current_rewards.append(1)

    def end_round(self):
        self.all_rewards.append(self.current_rewards)
        self.all_gradients.append(self.current_gradients)

    def debug(self):
        print("iteration {:3d} game {:5d} last_100_avg {:4.1f} max.score {:5d}".\
            format(self.iteration,self.game_number,self.last_100_avg_log[-1],self.max_score),end='\r')

    def logger(self,score):
        self.max_score=max(score,self.max_score)
        self.max_score_log.append(self.max_score)
        self.last_100.append(score)
        if len(self.last_100)>100:
            self.last_100=self.last_100[1:]
        self.last_100_avg_log.append(sum(self.last_100)/len(self.last_100))
        if self.iteration % self.save_per_iterations==0:
            saver.save(self.sess,self.model_checkpoints_path)


    def stateWrapper(self,state):
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