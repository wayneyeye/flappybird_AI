
from random import choice, random
class flappy_agent():
    def __init__(self,sample_interval=10):
        self.sample_interval=sample_interval
        self.discount_rate=0.5
        self.learning_rate=0.5
        self.learning_rate_min=0.05
        self.update_ct={}
        self.epsilon_0=0.99
        self.epsilon_decay=1
        
        self.iteration=0
        self.n_iteration=0
        self.state_dict={}
        self.rewards_list=[]
        self.discounted_rewards_list=[]
        self.state_list=[]
        self.action_list=[]
        self.last_score=0
        self.reward_rate=1
        self.punishment=1
        self.bonus=3
        self.highest_score=0
        self.last_100=[]
        self.last_100_average=0
        self.flap_rate=0.5
    
    def new_round(self):
        '''initialize'''
        self.last_100.append(self.last_score)
        if len(self.last_100)>100:
            self.last_100=self.last_100[1:]
        self.last_100_average=sum(self.last_100)/len(self.last_100)
        self.discounted_rewards_list=[]
        self.rewards_list=[]
        self.state_list=[]
        self.action_list=[]
        self.last_score=0
        self.iteration+=1
        
    def calculate_update(self):
        self.rewards_list[-1]-=self.punishment
        cumulated_reward=0
        r_index=len(self.rewards_list)-1
        # calculate discounted rewards
        for r in reversed(self.rewards_list):
            cumulated_reward=r+self.discount_rate*cumulated_reward
            self.discounted_rewards_list[r_index]=cumulated_reward
            r_index-=1
        
        # initialize unknown state-reward values and epsilon values
        for s in self.state_list:
            state_code=self.stateEncoder(s)
            if state_code not in self.state_dict:
                self.state_dict[state_code]={}
                self.state_dict[state_code][0]=0.0
                self.state_dict[state_code][1]=0.0
                self.update_ct[state_code]=0
        # update state_dict
        iter_limit=len(self.action_list)-1
        current_iter=0
        for s,a in zip(self.state_list,self.action_list):
            state_code=self.stateEncoder(s)
            if current_iter==iter_limit:
                break       
            next_state_code=self.stateEncoder(self.state_list[current_iter+1])
            # update Q(s,a)
            l_rate=max(self.learning_rate_min,self.learning_rate/(1+self.update_ct[state_code]*self.epsilon_decay))
            self.state_dict[state_code][a]=\
            (1-l_rate)*self.state_dict[state_code][a]+\
            l_rate*(self.rewards_list[current_iter]+\
                self.discount_rate*max(self.state_dict[next_state_code].values()))
            self.update_ct[state_code]+=1
            current_iter+=1

    def flap(self):
        if random()<self.flap_rate:
            return 1
        else:
            return 0

    def get_action(self,state):
        state_code=self.stateEncoder(state)
        if state_code in self.state_dict:
            if random()<=self.epsilon_0/(1+self.update_ct[state_code]*self.epsilon_decay) or self.state_dict[state_code][1]==self.state_dict[state_code][1]:
                return self.flap()
            else:
                if self.state_dict[state_code][1]>self.state_dict[state_code][0]:
                    return 1
                else:
                    return 0
        else:
            return self.flap()


    def next_step(self,state,action):
        self.state_list.append(state)
        self.action_list.append(action)
        current_reward=self.reward_rate
        if state["score_increase"]:
            current_reward+=self.bonus
        self.rewards_list.append(current_reward)
        self.discounted_rewards_list.append(current_reward)
        

    def debug(self):
        if self.last_score>=1 and self.iteration<100:
            print("rewards")
            print(self.rewards_list)
            print("discounted_rewards")
            print(self.discounted_rewards_list)
        # if self.iteration==500:
        #     print("Q-dicts")
        #     print_keys=list(self.state_dict.keys())
        #     print_keys=print_keys[-5:]
        #     for k in print_keys:
        #         print(self.state_dict[k])
        #     print("discounted_rewards")
        #     print(self.discounted_rewards_list)
        print("iteration: {:5d} highest_score: {:4d} states: {:6d} last_100_average:{:5.3f} ct_avg:{:3.3f} ct_min:{:2d}"\
            .format(self.iteration,self.highest_score,\
                len(self.state_dict),self.last_100_average,sum(self.update_ct.values())/len(self.update_ct.values()),min(self.update_ct.values())),end="\r")

    def stateEncoder(self,state):
        return str(state['player_pipe_Y'])+'#'+str(state['pipeRightPos'])
    
    def get_state(self,state,reduce_factor=20):
        playerLeftPos=state['playerx']
        pipeRightPos=state['lowerPipes'][0]['x']+52
        pipeY=state['lowerPipes'][0]['y']
        if pipeRightPos<=playerLeftPos:
            pipeRightPos=state['lowerPipes'][1]['x']+52
            pipeY=state['lowerPipes'][1]['y']
        playery=state['playery']
        playerVelY=state['playerVelY']
        self.highest_score=max(self.highest_score,state["score"])
        score_increase=state["score"]-self.last_score
        self.last_score=state["score"]
        return {'player_pipe_Y':(playery-pipeY)//reduce_factor,
                'pipeRightPos':pipeRightPos//reduce_factor,
                'playerVelY':playerVelY,
                'score_increase':score_increase,
                'crashed':state["crashed"]}

agent=flappy_agent()