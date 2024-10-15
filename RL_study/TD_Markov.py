#将时序差分的几种算法应用于简单的马尔可夫链例子中
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

S = ["s1", "s2", "s3", "s4", "s5"]
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往", "结束s5"]
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
    "s5-结束s5-s5": 1.0,
}
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
    "s5-结束s5": 0,
}



class MarkovChainEnv:
    def __init__(self, S, A, P, R, gamma=0.5):
        self.S = S  # 状态集合
        self.A = A  # 动作集合
        self.P = P  # 状态转移函数
        self.R = R  # 奖励函数
        self.gamma = gamma  # 折扣因子
        self.current_state = None

    def reset(self):
        """重置环境到初始状态"""
        self.current_state = np.random.choice(self.S[:-1])  # 随机选择一个除s5以外的状态作为起点
        return self.S.index(self.current_state)
    
    def pre_set(self, state):
        """设置环境到指定状态"""
        self.current_state = state
        return self.S.index(self.current_state)

    def step(self, action):
        """根据当前状态和动作返回下一个状态和奖励"""
        if self.current_state is None:
            raise ValueError("环境未初始化，请先调用reset方法。")
        print("当前状态：",self.current_state)
        print("动作：",action)
        state_action = f"{self.current_state}-{action}"
        reward = self.R.get(state_action, 0)
        
        rand = np.random.rand()
        temp = 0
        next_state = None
        done = False
        for s in self.S:
            temp += self.P.get(f"{state_action}-{s}", 0)
            if temp > rand:
                next_state = s
                break
        if next_state == self.S[-1]:
            done = True
        self.current_state = next_state
        print("下一状态：",self.current_state)
        return self.S.index(next_state), reward, done
    
def get_possible_actions(action, state, reward):
    """获取当前状态下的所有可能动作"""
    return [a for a in action if f"{state}-{a}" in reward]

def init_action_table(action, state, reward):
    """初始化动作表"""
    Q = np.zeros([len(state), len(action)])
    for s in range(len(state)):
        possible_actions = get_possible_actions(action, state[s], reward)
        n_actions = len(possible_actions)
        for a in possible_actions:
            Q[s, action.index(a)] = 1/n_actions
    return Q
    
    

class Sarsa:
    """ Sarsa算法 """
    def __init__(self, n_states, n_actions, epsilon, alpha, gamma,s_a_table):
        self.Q_table = np.zeros([n_states, n_actions])  # 初始化Q(s,a)表格
        self.s_a_table = s_a_table  # 记录所有可能的状态-动作对
        self.n_actions = n_actions  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        for i in range(n_states):
            for j in range(n_actions):
                if self.s_a_table[i, j] == 0:
                    self.Q_table[i, j] = -np.inf
        
        
    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        temp = 0
        if np.random.random() < self.epsilon:
            #print(1)
            for i in range(self.n_actions):
                #print("state:",state)
                temp += self.s_a_table[state, i] 
                if temp > np.random.rand() and self.s_a_table[state, i] != 0:
                    action = i
                    break  
            
        else:
            #print("Q_table:",self.Q_table[state])
            action = np.argmax(self.Q_table[state])
            
            #print(2)
        #print("takeaction动作：",action)
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_actions)]
        for i in range(self.n_actions):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


env = MarkovChainEnv(S, A, P, R)
#np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.5
s_a_table = init_action_table(A, S, R)
#print(s_a_table)
agent = Sarsa(len(S), len(A), epsilon, alpha, gamma, s_a_table)
num_episodes = 500  # 智能体在环境中运行的序列的数量

# for s in S:
#     print(get_possible_actions(A, s, R))

return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done = env.step(A[action])
                print("done:",done)
                next_action = agent.take_action(next_state)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on Custom Markov Chain Environment')
plt.show()

def print_agent(agent, env, action_meaning):
    for i in range(len(env.S)):
        a = agent.best_action(i)
        pi_str = ''
        for k in range(len(action_meaning)):
            # if a[k] > 0:
            #     pi_str += action_meaning[k]
            pi_str += action_meaning[k] if a[k] > 0 else 'o'
        print(f"State {env.S[i]}: {pi_str}")

action_meaning = A
print('Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning)  
print('Sarsa算法最终收敛得到的Q表为：')
print(agent.Q_table) 
    
    
# env = MarkovChainEnv(S, A, P, R)
# np.random.seed(0)
# env.reset()
# print("初始状态：", env.current_state)
# #action = env.get_possible_actions(env.current_state)
# #print(action)

# end_flag = False

# while not end_flag:
#     current_state = env.current_state
#     action = np.random.choice(env.get_possible_actions(env.current_state))
#     next_state, reward, end_flag = env.step(action)
#     print(current_state, action, next_state, reward, end_flag)
# env.pre_set("s1")
# #测试随机性，循环10次
# for i in range(10):
#     action = np.random.choice(env.get_possible_actions(env.current_state))
#     print(env.current_state, action)