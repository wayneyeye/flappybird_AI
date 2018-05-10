import time
from random import choice, random
class flappy_agent():
    def __init__(self,sample_interval=8):
        # agent sample interval (frames)
        self.sample_interval=sample_interval
        # model update parameters
        self.discount_rate=0.9
        self.learning_rate=0.4
        self.learning_rate_min=0.1
        self.epsilon_0=0.5
        self.epsilon_decay=0.001
        self.flap_rate=0.4 # random exploration
        # iteration count
        self.game_number=0
        self.iteration=0
        self.n_iteration=0
        self.steps=0
        self.last_100_steps=[]
        # q-table
        self.Q_table={}
        # sars recording
        self.rewards_list=[]
        self.discounted_rewards_list=[]
        self.state_list=[]
        self.action_list=[]
        # performance
        self.last_score=0
        self.start_time=time.time()
        
        self.highest_score=0
        self.last_100=[]
        self.last_100_avg_log=[]
        self.max_score=0
        self.max_score_log=[]
        self.last_100_steps_avg_log=[]
        # punishiment and bonus if crash or score up
        self.reward_rate=1
        self.punishment=0
        self.bonus=0
        
    
    def new_round(self):
        '''initialize'''
        self.discounted_rewards_list=[]
        # print(self.rewards_list)
        # print(self.action_list)
        self.rewards_list=[]
        self.state_list=[]
        self.action_list=[]
        self.last_score=0
        self.steps=0
        self.game_number+=1
        

    def update_model(self):
        self.rewards_list[-1]-=self.punishment
        cumulated_reward=0
        r_index=len(self.rewards_list)-1
        # calculate discounted rewards
        # for r in reversed(self.rewards_list):
        #     cumulated_reward=r+self.discount_rate*cumulated_reward
        #     self.discounted_rewards_list[r_index]=cumulated_reward
        #     r_index-=1
        
        # initialize unknown state-reward values and epsilon values
        for s in self.state_list:
            state_code=self.stateEncoder(s)
            if state_code not in self.Q_table:
                self.Q_table[state_code]={}
                self.Q_table[state_code][0]=0.0
                self.Q_table[state_code][1]=0.0
                self.Q_table[state_code]['ct']=0
        # update Q_table
        iter_limit=len(self.action_list)-1
        current_iter=0
        for s,a,r in zip(self.state_list,self.action_list,self.rewards_list):
            state_code=self.stateEncoder(s)
            if current_iter==iter_limit:
                max_Qsa_value=0
                # print(r)
            else:
            	next_state_code=self.stateEncoder(self.state_list[current_iter+1])
            	max_Qsa_value=max(self.Q_table[next_state_code][0],self.Q_table[next_state_code][1])       
            
            # update Q(s,a)
            l_rate=max(self.learning_rate_min,self.learning_rate/(1+self.Q_table[state_code]['ct']*self.epsilon_decay))
            self.Q_table[state_code][a]=(1-l_rate)*self.Q_table[state_code][a]+l_rate*(r+self.discount_rate*max_Qsa_value)
            self.Q_table[state_code]['ct']+=1
            current_iter+=1
        # if self.game_number==500:
        # 	print(l_rate)
        # 	print(self.Q_table)

    def end_round(self):
        self.iteration+=1



    def flap(self):
        if random()<self.flap_rate:
            return 1
        else:
            return 0

    def get_action(self,state):
        state_code=self.stateEncoder(state)
        if state_code in self.Q_table:
            if random()<=self.epsilon_0/(1+self.Q_table[state_code]['ct']*self.epsilon_decay) or self.Q_table[state_code][1]==self.Q_table[state_code][0]:
                return self.flap()
            else:
                if self.Q_table[state_code][1]>self.Q_table[state_code][0]:
                    return 1
                else:
                    return 0
        else:
            return self.flap()


    def next_step(self,state,action):
        self.steps+=1
        self.state_list.append(state)
        self.action_list.append(action)
        current_reward=self.reward_rate
        if state["score_increase"]:
            current_reward+=self.bonus
        self.rewards_list.append(current_reward)
        self.discounted_rewards_list.append(current_reward)
        

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
        score_increase=state["score"]-self.last_score
        self.last_score=state["score"]
        return {'player_pipe_Y':int((playery-pipeY)//40), # negative means the bird is higher than the pipe
                'pipeRightPos':int(pipeRightPos//48),
                'playerVelY':int(playerVelY//4),
                'playery':int(playery//40),
                'pipeY':int(pipeY//40),
                'score_increase':score_increase,
                'crashed':state["crashed"]}

    def logger(self,score):
        self.max_score=max(score,self.max_score)
        self.max_score_log.append(self.max_score)
        self.last_100_steps.append(self.steps)
        if len(self.last_100_steps)>100:
            self.last_100_steps=self.last_100_steps[1:]
        self.last_100_steps_avg_log.append(sum(self.last_100_steps)/len(self.last_100_steps))
        self.last_100.append(score)
        self.end_time=time.time()
        self.elapsed_time=self.end_time-self.start_time
        if len(self.last_100)>100:
            self.last_100=self.last_100[1:]
        self.last_100_avg_log.append(sum(self.last_100)/len(self.last_100))
        

agent=flappy_agent()