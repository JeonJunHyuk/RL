import numpy as np
import matplotlib.pyplot as plt
import time


class Environment:
    cliff = -3  # 나가면 -3점
    road = -1  # 기본 -1점
    goal = 1  # 도착하면 1점

    goal_position = [2, 2]  # 골 위치
    reward_list = [[road, road, road],  # 보상 리스트. 점수들.
                   [road, road, road],
                   [road, road, goal]]
    reward_list1 = [['road', 'road', 'road'],  # state 이름
                    ['road', 'road', 'road'],
                    ['road', 'road', 'goal']]

    def __init__(self):
        self.reward = np.asarray(self.reward_list)

    def move(self, agent, action):
        done = False  # True 되면 끝나게
        new_pos = agent.pos + agent.action[action]  # 기존 state에서 액션 수행해 새로운 state로 이동.

        # goal 이면 보상 받고 포지션 골로 이동하고 끝
        if self.reward_list1[agent.pos[0]][agent.pos[1]] == 'goal':
            reward = self.goal
            observation = agent.set_pos(agent.pos)
            done = True

        # cliff 면 보상 받고 포지션 이동하고 끝
        elif new_pos[0] < 0 or new_pos[0] >= self.reward.shape[0] or \
                new_pos[1] < 0 or new_pos[1] >= self.reward.shape[1]:
            reward = self.cliff
            observation = agent.set_pos(agent.pos)
            done = True

        # 그 외 이동하면 그 새로운 곳으로.
        else:
            observation = agent.set_pos(new_pos)
            reward = self.reward[observation[0], observation[1]]

        return observation, reward, done


class Agent:
    action = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])  # 가능한 액션들
    select_action_pr = np.array([0.25, 0.25, 0.25, 0.25])  # 각 액션 선택할 확률

    def __init__(self, initial_position):
        self.pos = initial_position

    def set_pos(self, position):  # 새로운 position 저장
        self.pos = position
        return self.pos

    def get_pos(self):  # 현재 위치 불러오기
        return self.pos


def state_value_function(env, agent, G, max_step, now_step):
    gamma = 0.9  # 감가율 0.9
    if env.reward_list1[agent.pos[0]][agent.pos[1]] == 'goal':  # goal 도착하면 env.goal=1 반환
        return env.goal

    if max_step == now_step:  # 마지막 스텝이면
        pos1 = agent.get_pos()  # 현재 위치를 가져와서 pos1에 저장
        for i in range(len(agent.action)):  # 액션 개수(4)만큼
            agent.set_pos(pos1)  # agent 를 pos1(현재 위치)에 놓고
            observation, reward, done = env.move(agent, i)  # 액션 종류별로 이동해 agent 위치 저장
            G += agent.select_action_pr[i] * reward  # 액션을 선택할 확률에 reward 곱해서 G에 차곡차곡 쌓음.

        return G

    else:  # 스텝 진행 중이면
        pos1 = agent.get_pos()  # 현재 위치 pos1에 저장.

        for i in range(len(agent.action)):  # 모든 액션 4개 다 돌아가면서 할 건데
            observation, reward, done = env.move(agent, i)  # 움직여서 결과 저장
            G += agent.select_action_pr[i] * reward  # G도 계산해서 차곡차곡
            if done == True:  # 끝나거나 밖으로 나가면
                if observation[0] < 0 or \
                        observation[0] >= env.reward.shape[0] or \
                        observation[1] < 0 or \
                        observation[1] >= env.reward.shape[1]:
                    agent.set_pos(pos1)  # 다시 pos1으로 돌아가기

            next_v = state_value_function(env, agent, 0, max_step,
                                          now_step + 1)  # now_step +1 해서 또 state_value_function 계산
            G += agent.select_action_pr[i] * gamma * next_v  # 감가율 곱해서 G에 차곡차곡.

            agent.set_pos(pos1)  # 다시 pos1으로 돌아가기
        return G


def action_value_function(env, agent, act, G, max_step, now_step):
    gamma = 0.9
    if env.reward_list1[agent.pos[0]][agent.pos[1]] == "goal":  # goal이면 1점 리턴
        return env.goal

    if (max_step == now_step):  # 마지막 스텝일 땐 보상만 계산
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act] * reward
        return G

    else:
        pos1 = agent.get_pos()
        observation, reward, done = env.move(agent, act)
        G += agent.select_action_pr[act] * reward

        if done == True:
            if observation[0] < 0 or \
                    observation[0] >= env.reward.shape[0] or \
                    observation[1] < 0 or \
                    observation[1] >= env.reward.shape[1]:
                agent.set_pos(pos1)

        pos1 = agent.get_pos()

        for i in range(len(agent.action)):
            agent.set_pos(pos1)
            next_v = action_value_function(env, agent, i, 0, max_step, now_step + 1)
            G += agent.select_action_pr[i] * gamma * next_v
        return G


def show_v_table(v_table, env):
    for i in range(env.reward.shape[0]):  # 세로 칸 수가 3개. i 로 돌아감.
        print("+-----------------" * env.reward.shape[1], end="")  # 가로 칸 수 3개
        print("+")
        for k in range(3):  # 한 칸에 들어가는 표현을 세 줄로.
            print("|", end="")
            for j in range(env.reward.shape[1]):  # 가로 세 칸 표현.
                if k == 0:
                    print("                 |", end="")
                if k == 1:
                    print("   {0:8.2f}      |".format(v_table[i, j]), end="")
                if k == 2:
                    print("                 |", end="")
            print()
    print("+-----------------" * env.reward.shape[1], end="")
    print("+")


env = Environment()

agent = Agent([0, 0])

max_step_number = 8

time_len = []

for max_step in range(max_step_number):  # max_step 1-8 마다 상태가치함수 계산 + 테이블 보여주기
    v_table = np.zeros((env.reward.shape[0], env.reward.shape[1]))
    start_time = time.time()

    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            agent.set_pos([i, j])
            v_table[i, j] = state_value_function(env, agent, 0, max_step, 0)

    time_len.append(time.time() - start_time)
    print("max_step_number = {} total_time = {}(s)".format(max_step, np.round(time.time() - start_time, 2)))

    show_v_table(np.round(v_table, 2), env)

plt.plot(time_len, 'o-k')
plt.xlabel('max_down')
plt.ylabel('time(s)')
plt.legend()
plt.show()


def show_q_table(q_table, env):
    for i in range(env.reward.shape[0]):
        print("+-----------------" * env.reward.shape[1], end='')
        print("+")
        for k in range(3):
            print('|', end='')
            for j in range(env.reward.shape[1]):
                if k == 0:
                    print("{0:10.2f}      |".format(q_table[i, j, 0]), end='')
                if k == 1:
                    print("{0:6.2f}    {1:6.2f} |".format(q_table[i, j, 3], q_table[i, j, 1]), end='')
                if k == 2:
                    print("{0:10.2f}      |".format(q_table[i, j, 2]), end="")
            print()
    print('+-----------------' * env.reward.shape[1], end="")
    print("+")


def show_q_table_arrow(q_table, env):
    for i in range(env.reward.shape[0]):
        print("+-----------------" * env.reward.shape[1], end="")
        print("+")
        for k in range(3):
            print("|", end="")
            for j in range(env.reward.shape[1]):
                if k == 0:
                    if np.max(q[i, j, :]) == q[i, j, 0]:
                        print("        ↑       |", end="")
                    else:
                        print("                 |", end="")
                if k == 1:
                    if np.max(q[i, j, :]) == q[i, j, 1] and np.max(q[i, j, :]) == q[i, j, 3]:
                        print("      ←  →     |", end="")
                    elif np.max(q[i, j, :]) == q[i, j, 1]:
                        print("          →     |", end="")
                    elif np.max(q[i, j, :]) == q[i, j, 3]:
                        print("      ←         |", end="")
                    else:
                        print("                 |", end="")
                if k == 2:
                    if np.max(q[i, j, :]) == q[i, j, 2]:
                        print("        ↓       |", end="")
                    else:
                        print("                 |", end="")
            print()
    print("+-----------------" * env.reward.shape[1], end="")
    print("+")


for max_step in range(max_step_number):  # max_step 1-8 마다 행동가치함수 계산 + 테이블 보여주기
    print("max_step = {}".format(max_step))
    q_table = np.zeros((env.reward.shape[0], env.reward.shape[1], len(agent.action)))  # q_table은 3차원.
    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            for action in range(len(agent.action)):
                agent.set_pos([i, j])
                q_table[i, j, action] = action_value_function(env, agent, action, 0, max_step, 0)

    q = np.round(q_table, 2)
    print("Q-table")
    show_q_table(q, env)
    print("High actions Arrow")
    show_q_table_arrow(q, env)
    print()

# 반복 정책 평가 60p. 바로 다음 상태만 이용해 상태가치 계산
env = Environment()
agent = Agent([0,0])
gamma = 0.9

v_table = np.zeros((env.reward.shape[0], env.reward.shape[1]))
k = 1

import copy

show_v_table(np.round(v_table,2),env)
start_time = time.time()
# k 번째와 k-1 번째 가치 차이가 delta 아래로 내려갈 때까지 반복
while(True):
    delta = 0

    temp_v = copy.deepcopy(v_table)

    for i in range(env.reward.shape[0]):
        for j in range(env.reward.shape[1]):
            G = 0

            for action in range(len(agent.action)):
                agent.set_pos([i,j])
                observation, reward, done = env.move(agent, action)

                G+= agent.select_action_pr[action] * (reward + gamma*v_table[observation[0],observation[1]])

            v_table[i,j]=G

    delta = np.max([delta, np.max(np.abs(temp_v - v_table))])

    end_time = time.time()
    print("V{0}(S) : k = {1:3d}    delta = {2:0.6f} total_time = {3}".format(
        k, k, delta, np.round(end_time - start_time), 2))
    show_v_table(np.round(v_table,2),env)
    k+=1

    if delta<0.00001:
        break

end_time = time.time()
print("total_time = {}".format(end_time-start_time))