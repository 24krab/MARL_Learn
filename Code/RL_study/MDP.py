#Markov决策过程（MDP）
import numpy as np
import copy
np.random.seed(0)


S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
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
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子 初始0.5
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}
#策略3
Pi_3 = {
    "s1-保持s1": 0.1,
    "s1-前往s2": 0.9,
    "s2-前往s1": 0.1,
    "s2-前往s3": 0.9,
    "s3-前往s4": 0.1,
    "s3-前往s5": 0.9,
    "s4-前往s5": 0.9,
    "s4-概率前往": 0.1,
}


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + '-' + str2

def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解,states_num是MRP的状态数 '''
    rewards = np.array(rewards).reshape((-1, 1))  #将rewards写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P),
                   rewards)
    return value

#gamma = 0.5
# 转化后的MRP的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
print("MDP中每个状态价值分别为\n", V)


def sample(MDP, Pi, timestep_max, number):
    ''' 采样函数,策略Pi,限制最长时间步timestep_max,总共采样序列数number '''
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]  # 随机选择一个除s5以外的状态s作为起点
        # 当前状态为终止状态或者时间步太长时,一次采样结束
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            # 在状态s下根据策略选择动作
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)#Pi.get(join(s, a_opt), 0)返回键join(s, a_opt)对应的值,如果没有则返回0
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            # 根据状态转移概率得到下一个状态s_next
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))  # 把（s,a,r,s_next）元组放入序列中
            s = s_next  # s_next变成当前状态,开始接下来的循环
        episodes.append(episode)
    return episodes


# # 采样5次,每个序列最长不超过20步
# episodes = sample(MDP, Pi_3, 20, 5)
# print('第一条序列\n', episodes[0])
# print('第二条序列\n', episodes[1])
# print('第五条序列\n', episodes[4])

# # 对所有采样序列计算所有状态的价值
# def MC(episodes, V, N, gamma):
#     for episode in episodes:
#         G = 0
#         for i in range(len(episode) - 1, -1, -1):  #一个序列从后往前计算
#             (s, a, r, s_next) = episode[i]
#             G = r + gamma * G
#             N[s] = N[s] + 1
#             V[s] = V[s] + (G - V[s]) / N[s]


# timestep_max = 20
# # 采样1000次,可以自行修改
# episodes = sample(MDP, Pi_3, timestep_max, 1000)
# gamma = 0.5
# V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
# N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
# MC(episodes, V, N, gamma)
# print("使用蒙特卡洛方法计算MDP的状态价值为\n", V)


# def occupancy(episodes, s, a, timestep_max, gamma):
#     ''' 计算状态动作对（s,a）出现的频率,以此来估算策略的占用度量 '''
#     rho = 0
#     total_times = np.zeros(timestep_max)  # 记录每个时间步t各被经历过几次
#     occur_times = np.zeros(timestep_max)  # 记录(s_t,a_t)=(s,a)的次数
#     for episode in episodes:
#         for i in range(len(episode)):
#             (s_opt, a_opt, r, s_next) = episode[i]
#             total_times[i] += 1
#             if s == s_opt and a == a_opt:
#                 occur_times[i] += 1
#     for i in reversed(range(timestep_max)):
#         if total_times[i]:
#             rho += gamma**i * occur_times[i] / total_times[i]
#     return (1 - gamma) * rho

# gamma = 0.5
# timestep_max = 1000

# episodes_1 = sample(MDP, Pi_1, timestep_max, 1000)
# episodes_2 = sample(MDP, Pi_2, timestep_max, 1000)
# episodes_3 = sample(MDP, Pi_3, timestep_max, 1000)
# # rho_1 = occupancy(episodes_1, "s4", "概率前往", timestep_max, gamma)
# # rho_2 = occupancy(episodes_2, "s4", "概率前往", timestep_max, gamma)
# # rho_3 = occupancy(episodes_3, "s4", "概率前往", timestep_max, gamma)
# rho_1 = occupancy(episodes_1, "s4", "前往s5", timestep_max, gamma)
# rho_2 = occupancy(episodes_2, "s4", "前往s5", timestep_max, gamma)
# rho_3 = occupancy(episodes_3, "s4", "前往s5", timestep_max, gamma)


# print(rho_1, rho_2, rho_3)


#使用动态规划寻找最优策略
#策略迭代算法
class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, MDP, init_Pi, theta, gamma):
        self.S, self.A, self.P, self.R, self.gamma = MDP
        self.v = {s: 0 for s in self.S}  # 初始化价值为0
        #print(self.v)
        self.pi = init_Pi  # 初始化策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  # 策略评估
        cnt = 0  # 计数器
        while 1:
            delta = 0
            new_v = {s: 0 for s in self.S}
            for s in self.S:
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a_opt in self.A:
                    qsa = 0
                    if self.pi.get(join(s, a_opt), 0) > 0:
                        for s_opt in self.S:
                            if self.P.get(join(join(s, a_opt), s_opt), 0) > 0:
                                qsa += self.P[join(join(s, a_opt), s_opt)] * (
                                    self.R.get(join(s, a_opt), 0) + self.gamma * self.v[s_opt])
                                #print("s:",s,"a_opt:",a_opt,"qsa:",qsa)
                        qsa_list.append(self.pi.get(join(s, a_opt), 0) * qsa)
                        #print("cnt:",cnt+1,"s:",s,"a_opt:",a_opt,"qsa:",qsa)#检测代码
                new_v[s] = sum(qsa_list)
                #print("cnt:",cnt+1,"s:",s,"new_v[s]:",new_v[s])
                delta = max(delta, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if delta < self.theta:
                print(self.v)
                print("策略评估迭代次数为：", cnt+1)
                break
            cnt += 1
            if cnt > 1000:
                print("err")
                break

    def policy_improvement(self):  # 策略提升
        for s in self.S:
            print("当前状态：", s)
            if s == "s5":
                continue
            qsa_list = []
            s_a_opt_list = []
            for a_opt in self.A:
                qsa = 0
                if self.pi.get(join(s, a_opt), 0) > 0:
                    for s_opt in self.S:
                        if self.P.get(join(join(s, a_opt), s_opt), 0) > 0:
                            qsa += self.P[join(join(s, a_opt), s_opt)] * (
                                self.R.get(join(s, a_opt), 0) + self.gamma * self.v[s_opt])
                    s_a_opt_list.append(join(s, a_opt))        
                    qsa_list.append(qsa) 
            print(qsa_list) 
            maxq = max(qsa_list)

            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            q = 1.0/cntq
            #print("s:",s,"maxq:",maxq,"cntq:",cntq,"q:",q)
            for s_a_opt in s_a_opt_list:
                if self.pi.get(s_a_opt, 0) > 0 and qsa_list[s_a_opt_list.index(s_a_opt)] == maxq:
                    self.pi[s_a_opt] = q
                else:
                    self.pi[s_a_opt] = 0
            
        print("策略提升完成")
        print(self.pi)
        return self.pi
            
    def policy_iteration(self):
        cnt = 0
        while 1:
            #self.policy_evaluation()
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            if old_pi == new_pi: 
                print("策略迭代次数为：", cnt+1)
                break
            cnt += 1
            if cnt > 1000:
                print("err")
                break
        return self.v, self.pi
    
    
agent = PolicyIteration(MDP, Pi_1, 1e-6, 0.9)
agent.policy_iteration()

#价值迭代算法
class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, MDP, init_pi, theta, gamma):
        self.S, self.A, self.P, self.R, self.gamma = MDP
        self.v = {s: 0 for s in self.S}  # 初始化价值为0
        self.theta = theta  # 价值评估收敛阈值
        self.gamma = gamma  # 折扣因子 
        self.pi = init_pi  # 初始化策略

    def value_iteration(self):
        cnt = 0  # 计数器
        while 1:
            delta = 0
            new_v = {s: 0 for s in self.S}
            for s in self.S:
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a_opt in self.A:
                    qsa = 0
                    for s_opt in self.S:
                        qsa += self.P.get(join(join(s, a_opt), s_opt), 0) * (
                            self.R.get(join(s, a_opt), 0) + self.gamma * self.v[s_opt])
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                delta = max(delta, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if delta < self.theta:
                print(self.v)
                print("价值迭代次数为：", cnt+1)
                break
            cnt += 1
            if cnt > 1000:
                print("err")
                break
        self.pi = self.get_policy()
        print("最优策略为：", self.pi)
        return self.v

    def get_policy(self):
        for s in self.S:
            if s == "s5":
                continue
            qsa_list = []
            s_a_opt_list = []
            # for a_opt in self.A:
            #     qsa = 0
            #     for s_opt in self.S:
            #         qsa += self.P.get(join(join(s, a_opt), s_opt), 0) * (
            #             self.R.get(join(s, a_opt), 0) + self.gamma * self.v[s_opt])
            #     qsa_list.append(qsa)
            # maxq = max(qsa_list)
            # cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
            for a_opt in self.A:
                qsa = 0
                if self.pi.get(join(s, a_opt), 0) > 0:
                    for s_opt in self.S:
                        if self.P.get(join(join(s, a_opt), s_opt), 0) > 0:
                            qsa += self.P[join(join(s, a_opt), s_opt)] * (
                                self.R.get(join(s, a_opt), 0) + self.gamma * self.v[s_opt])
                    s_a_opt_list.append(join(s, a_opt))        
                    qsa_list.append(qsa) 
            print(qsa_list) 
            maxq = max(qsa_list)

            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            q = 1.0/cntq
            #print("s:",s,"maxq:",maxq,"cntq:",cntq,"q:",q)
            for s_a_opt in s_a_opt_list:
                if self.pi.get(s_a_opt, 0) > 0 and qsa_list[s_a_opt_list.index(s_a_opt)] == maxq:
                    self.pi[s_a_opt] = q
                else:
                    self.pi[s_a_opt] = 0
        return self.pi
    
agent = ValueIteration(MDP, Pi_1,1e-6, 0.5)
agent.value_iteration()
