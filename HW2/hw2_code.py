#!/usr/bin/env python
# coding: utf-8

# # Question 2: GridWorld with solving Bellman equations 

# In[21]:


import numpy as np

# discount rate gamma
discount_rate=0.9

# constant variables regarding state environments
num_rows=5

num_cols=5
num_states=25

# coefficient matrix and constant vector for solving system of linear equations
grid_world_state_coefficients=np.zeros((num_states, num_states))
grid_world_constants=np.zeros(num_states)

# possible action sequences in any given state
actions={'north':(-1, 0), 'south':(1, 0), 'east':(0, 1), 'west':(0, -1)}

# policy for each action action taken (in this case equiprobable)
pi=0.25

# Given a current state and an action taken, returns the next state and the reward obtained
def next_state_and_reward(current_state, current_action):
    reward=0
    next_state=[0, 0]
        
    # if current state is state A, then reward is 10 and next state is A' regardless of the action taken
    if(current_state[0]==0 and current_state[1]==1):
        reward=10
        next_state=[4, 1]
        return reward, next_state
    
    # if current state is state B, then reward is 5 and next state is B' regardless of the action taken
    elif(current_state[0]==0 and current_state[1]==3):
        reward=5
        next_state=[2, 3]
        return reward, next_state
    
    # if next state that is reached based on action taken is within the grid world, then return next state with reward 0
    if((current_state[0]+current_action[0])>=0 and (current_state[0]+current_action[0])<=4 and (current_state[1]+current_action[1])>=0 and (current_state[1]+current_action[1])<=4):
        reward=0
        next_state=[current_state[0]+current_action[0], current_state[1]+current_action[1]]        
        return reward, next_state
    
    # if next state that is reached based on action taken is outside gridworld, then return next state as current state with reward -1
    else:
        reward=-1
        next_state=[current_state[0], current_state[1]]
        return reward, next_state

# create coefficient and constant matrices for solving linear system of equations
for state in range(num_states):
    # The coefficient for a state s in its own state equation (row s) will be: 
    # c=1-n_r*(pi*prob(s,r|s,a)*gamma)
    #
    # The coefficient for a different state s' in a different state's state equation (row s) will be:
    # c'=1-pi*prob(s',r|s,a)*gamma
    #
    # The constant term for an equation for state s (row s) will be:
    # const=pi*(r1+r2+r3+r4)
    # where r1 is the reward from action north, r2 is from action south, r3 from action east and r4 from action west
    grid_world_state_coefficients[state][state]+=1
    
    for action in actions:
        reward, next_state=next_state_and_reward([state%num_rows, state//num_rows], actions[action])
        grid_world_state_coefficients[state, next_state[0]+next_state[1]*num_rows]-=pi*discount_rate
        grid_world_constants[state]+=pi*reward

# Solve system of linear equations to get the values of each of the state-action value functions
value_function=np.linalg.solve(grid_world_state_coefficients, grid_world_constants)
value_function.round(1).reshape((num_rows, num_cols)).transpose()
    


# # Question 4: Optimal State-Value Function and Optimal Policy 

# In[22]:


import numpy as np
from scipy import optimize

# discount rate gamma
discount_rate=0.9

# constant variables regarding state environments
num_rows=5
num_cols=5
num_states=25
num_actions=4

# coefficient matrix and constant vector for solving non linear system of equations for optimal action-value function
optimal_action_coefficients=np.zeros((num_actions*num_states, num_states))
optimal_action_constants=np.zeros(num_actions*num_states)

# possible action sequences in any given state
actions={'north':(-1, 0), 'south':(1, 0), 'east':(0, 1), 'west':(0, -1)}
mapping={0:'W', 1:'S', 2:'E', 3:'N'}

# policy for each action action taken (in this case equiprobable)
pi=0.25

# Given a current state and an action taken, returns the next state and the reward obtained
def next_state_and_reward(current_state, current_action):
    reward=0
    next_state=[0, 0]
        
    # if current state is state A, then reward is 10 and next state is A' regardless of the action taken
    if(current_state[0]==0 and current_state[1]==1):
        reward=10
        next_state=[4, 1]
        return reward, next_state
    
    # if current state is state B, then reward is 5 and next state is B' regardless of the action taken
    elif(current_state[0]==0 and current_state[1]==3):
        reward=5
        next_state=[2, 3]
        return reward, next_state
    
    # if next state that is reached based on action taken is within the grid world, then return next state with reward 0
    if((current_state[0]+current_action[0])>=0 and (current_state[0]+current_action[0])<=4 and (current_state[1]+current_action[1])>=0 and (current_state[1]+current_action[1])<=4):
        reward=0
        next_state=[current_state[0]+current_action[0], current_state[1]+current_action[1]]        
        return reward, next_state
    
    # if next state that is reached based on action taken is outside gridworld, then return next state as current state with reward -1
    else:
        reward=-1
        next_state=[current_state[0], current_state[1]]
        return reward, next_state

# create coefficient and constant matrices for solving non linear system of equations
for state in range(num_states):
    # we need to solve non linear equations of the form: Ax>=b since
    # A is the coefficient matrix and b is the constant matrix and the bellman optimality equations are all of the form
    # v_pi*(s)=max(.), therefore for each action we will have 4 such equations and hence there will be
    # totally 25*4 = 100 equations
    grid_world_state_coefficients[state][state]+=1
    
    for action_index, action in enumerate(actions):
        reward, next_state=next_state_and_reward([state%num_rows, state//num_rows], actions[action])
        optimal_action_coefficients[num_actions*state+action_index, state]+=1
        optimal_action_coefficients[num_actions*state+action_index, next_state[0]+next_state[1]*num_rows]-=discount_rate
        optimal_action_constants[num_actions*state+action_index]+=reward

        
# Solve system of non linear equations to get the values of each of the state-action value functions
optimal_value_function=optimize.linprog(np.ones(num_states), -optimal_action_coefficients, -optimal_action_constants)
optimal_value_function=np.asarray(optimal_value_function.x).round(1)

print('The optimal value function is:')
print(np.asarray(optimal_value_function.reshape((num_rows, num_cols)).transpose()))

print()
print('The optimal policy is:')

# To find optimal policy from the optimal value function
for state in range(num_states):
    
    # For each state, find next state from all actions
    q_pi_values=np.zeros(num_actions)
    for action_index, action in enumerate(actions):
        reward, next_state=next_state_and_reward([state//num_rows, state%num_rows], actions[action])
        q_pi_values[action_index]=optimal_value_function[next_state[0]+next_state[1]*num_rows]
    
    max_action=np.max(q_pi_values)
    
    for q_pi_value_index, q_pi_value in enumerate(q_pi_values):
        if(q_pi_value==max_action):
            print(mapping[q_pi_value_index], end='')
    if((state+1)%num_rows==0):
        print()
    else:
        print(',', end='')


# # Question 6: Policy Iteration

# In[18]:


# Policy Iteration

import sys
from copy import deepcopy

# parameters for MDP
num_rows=4
num_cols=4
num_states=16
grid_world=np.zeros(num_states)
num_actions=4

# mappings for optimal actions from value function
actions={'north': [-1, 0], 'south': [1, 0], 'east': [0, 1], 'west': [0, -1]}
mapping={0:'W', 1:'S', 2:'E', 3:'N'}

# theta constraint
theta=0.001

# policy matrix containing optimal actions for each state
pi_matrix=np.zeros((num_states, num_actions))
pi_matrix.fill(1/num_rows)

# state value function
state_value_function=np.zeros(num_states)

# return reward
def get_reward(state, action):
    return -1


# get next state and reward given current state and action
def get_next_state(state, action):
    i, j=state//num_rows, state%num_rows
    
    if(i==0 and j==0):
        next_state=[0, 0]
        return next_state[0]*num_rows+next_state[1]
    
    if(i==num_rows-1 and j==num_cols-1):
        next_state=[num_rows-1, num_cols-1]
        return next_state[0]*num_rows+next_state[1]
    
    if(i+action[0]>=0 and i+action[0]<=3 and j+action[1]>=0 and j+action[1]<=3):
        next_state=[i+action[0], j+action[1]]
        return next_state[0]*num_rows+next_state[1]
    
    next_state=[i, j]
    return next_state[0]*num_rows+next_state[1]
  
    
# policy iteration loop

while(True):
    
    # Policy evaluation
    while(True):
        delta=0
        for state in range(num_states):
            v=state_value_function[state]
            new_state_value=0
            
            # terminal states
            if(state==0 or state==num_states-1):
                continue
            
            # get updated state value function
            for action_index, action in enumerate(actions):
                new_state_value+=pi_matrix[state][action_index]*(get_reward(state, action)+state_value_function[get_next_state(state, actions[action])])
            
            delta=max(delta, abs(v-new_state_value))
            state_value_function[state]=new_state_value
        
        # check if updated value function difference is miniscule
        if(delta<theta):
            break
        print('delta: ', delta)
    print('Policy evaluation update:')
    print(state_value_function.round(2))
        
    # Policy improvement
    
    policy_stable=True
    for state in range(num_states):
        
        # terminal states
        if(state==0 or state==num_states-1):
            continue
        
        old_action=deepcopy(pi_matrix[state])
        
        max_action_value=-sys.maxsize-1
        
        # get optimal action's value function
        for action_index, action in enumerate(actions):
            current_action_value=get_reward(state, action)+state_value_function[get_next_state(state, actions[action])]
            
            if(current_action_value>max_action_value):
                max_action_value=current_action_value
        
        # get optimal actions for each state stochastically
        best_actions=[]
        
        for action_index, action in enumerate(actions):
            current_action_value=get_reward(state, action)+state_value_function[get_next_state(state, actions[action])]
            if(current_action_value==max_action_value):
                best_actions.append(action_index)
        all_actions=[0, 1, 2, 3]
        for other_action in all_actions:
            if(other_action in best_actions):
                pi_matrix[state][other_action]=1/len(best_actions)
            else:
                pi_matrix[state][other_action]=0
        # check if updated policy is stable or not
        for iterate in range(len(old_action)):
            if(old_action[iterate]!=pi_matrix[state][iterate]):
                policy_stable=False
                break
    
    print('Policy improvement update:')
    print(pi_matrix)
    if(policy_stable):
        break
print()
print('Final optimal policy value function:')
print(state_value_function.reshape((num_rows, num_cols)))
print()
print('Final optimal policy')

# show optimal policy

for s_index, state_policy in enumerate(pi_matrix):
    if(s_index==0):
        print('-,', end='')
        continue
    if(s_index==num_states-1):
        print('-', end='')
        continue
        
    max_s=max(state_policy)
    
    for val in range(len(state_policy)):
        if(state_policy[val]==max_s):
            print(mapping[val], end='')
            
    print(',', end='')

    if(s_index!=0 and (s_index+1)%num_rows==0):
        print()


# # Question 6: Value Iteration

# In[20]:


# Value Iteration

import sys
from copy import deepcopy

# initialise parameters for MDP

num_rows=4
num_cols=4
num_states=16
grid_world=np.zeros(num_states)
num_actions=4

# mappings for finding optimal actions from state value function 

actions={'north': [-1, 0], 'south': [1, 0], 'east': [0, 1], 'west': [0, -1]}
mapping={0:'W', 1:'S', 2:'E', 3:'N'}

theta=0.001

# policy matrix containing optimal actions for each state

pi_matrix=np.zeros((num_states, num_actions))
pi_matrix.fill(1/num_rows)

# state value function for each state

state_value_function=np.zeros(num_states)

# return reward
def get_reward(state, action):
    return -1

# return next state and reward given current state and action
def get_next_state(state, action):
    i, j=state//num_rows, state%num_rows
    
    if(i==0 and j==0):
        next_state=[0, 0]
        return next_state[0]*num_rows+next_state[1]
    
    if(i==num_rows-1 and j==num_cols-1):
        next_state=[num_rows-1, num_cols-1]
        return next_state[0]*num_rows+next_state[1]
    
    if(i+action[0]>=0 and i+action[0]<=3 and j+action[1]>=0 and j+action[1]<=3):
        next_state=[i+action[0], j+action[1]]
        return next_state[0]*num_rows+next_state[1]
    
    next_state=[i, j]
    return next_state[0]*num_rows+next_state[1]
    

# value iteration steps
    
# Policy evaluation
while(True):
    delta=0
    for state in range(num_states):
        v=state_value_function[state]
        new_state_value=-sys.maxsize-1

        # terminal states
        if(state==0 or state==num_states-1):
            continue

        # get max state value action considering all actions    
        for action_index, action in enumerate(actions):
            new_state_value=max(get_reward(state, action)+state_value_function[get_next_state(state, actions[action])], new_state_value)

        delta=max(delta, abs(v-new_state_value))
        state_value_function[state]=new_state_value
    if(delta<theta):
        break
    print('delta: ', delta)

    print('Policy evaluation update:')
    print(state_value_function.round(2))
        
# Policy improvement
for state in range(num_states):

    # terminal states
    if(state==0 or state==num_states-1):
        continue

    old_action=deepcopy(pi_matrix[state])

    max_action_value=-sys.maxsize-1

    # get optimal actions' state value function
    for action_index, action in enumerate(actions):
        current_action_value=get_reward(state, action)+state_value_function[get_next_state(state, actions[action])]

        if(current_action_value>max_action_value):
            max_action_value=current_action_value

            
    # get optimal actions for each state stochastically 
    best_actions=[]

    for action_index, action in enumerate(actions):
        current_action_value=get_reward(state, action)+state_value_function[get_next_state(state, actions[action])]
        if(current_action_value==max_action_value):
            best_actions.append(action_index)
    all_actions=[0, 1, 2, 3]
    for other_action in all_actions:
        if(other_action in best_actions):
            pi_matrix[state][other_action]=1/len(best_actions)
        else:
            pi_matrix[state][other_action]=0

print('Policy improvement update:')
print(pi_matrix)

print()
print('Final optimal policy value function:')
print(state_value_function.reshape((num_rows, num_cols)))
print()
print('Final optimal policy')

# get optimal policy from mapping from state value function

for s_index, state_policy in enumerate(pi_matrix):
    if(s_index==0):
        print('-,', end='')
        continue
    if(s_index==num_states-1):
        print('-', end='')
        continue
        
    max_s=max(state_policy)
    
    for val in range(len(state_policy)):
        if(state_policy[val]==max_s):
            print(mapping[val], end='')
            
    print(',', end='')

    if(s_index!=0 and (s_index+1)%num_rows==0):
        print()


# # Question 7: Jack's Car Rental part 1

# In[ ]:


# Jacks' Car rental, eg. 4.2 reconstruction

import sys
from copy import deepcopy
import math
import matplotlib.pyplot as plt

# initialise parameters of MDP
num_cars_A=21
num_cars_B=21
num_states=num_cars_A*num_cars_B
grid_world=np.zeros(num_states)
num_actions=11
theta=1

# expectations of poisson distributions for the requests and returns for each location 
expected_requests_A=3
expected_requests_B=4
expected_returns_A=3
expected_returns_B=2

# max number of rentals or returns in a single day possible
max_number_cars_rented_or_returned=11
# discount rate
gamma=0.9

# rewards as specified
rent_reward=10
moving_reward=-2

# define the poisson distribution value for a given lambda and n
def poisson(n, lambda_val):
    return np.exp(-1*lambda_val)*np.power(lambda_val, n)/np.math.factorial(n)

# method to return the expected return from a particular state given a chosen action
def get_state_value(car_a, car_b, pi_matrix, state_value_function):
    action=pi_matrix[car_a][car_b]
    moved_cars_a=car_a-action
    expected_return=moving_reward*abs(pi_matrix[car_a][car_b])
    moved_cars_b=car_b+action
    cars_in_a=max(0, min(moved_cars_a, num_cars_A-1))
    cars_in_b=max(0, min(moved_cars_b, num_cars_B-1))
    
    for request_sample_a in range(max_number_cars_rented_or_returned):
        for request_sample_b in range(max_number_cars_rented_or_returned):
            for return_sample_a in range(max_number_cars_rented_or_returned):
                for return_sample_b in range(max_number_cars_rented_or_returned):
                    final_prob=poisson(request_sample_a, expected_requests_A)
                    final_prob*=poisson(request_sample_b, expected_requests_B)
                    final_prob*=poisson(return_sample_a, expected_returns_A)
                    final_prob*=poisson(return_sample_b, expected_returns_B)
                    
                    rentals_A=min(cars_in_a, request_sample_a)
                    rentals_B=min(cars_in_b, request_sample_b)
                    
                    cars_in_a_end=int(min(cars_in_a+return_sample_a-rentals_A, num_cars_A-1))
                    cars_in_b_end=int(min(cars_in_b+return_sample_b-rentals_B, num_cars_B-1))
                    
                    reward=rent_reward*(rentals_A+rentals_B)
                    
                    expected_return+=final_prob*(reward+gamma*state_value_function[cars_in_a_end][cars_in_b_end])

    return expected_return

# policy matrix containing optimal actions for each state
pi_matrix=np.zeros((num_cars_A, num_cars_B))
# state value function for each state
state_value_function=np.zeros((num_cars_A, num_cars_B))

possible_actions=np.zeros(num_actions)
start_action=-5
for possible_action in range(num_actions):
    possible_actions[possible_action]=start_action
    start_action+=1


# Policy iteration    
while(True):
    
    # policy evaluation
    while(True):
        delta=0
        
        for car_a in range(num_cars_A):
            for car_b in range(num_cars_B):
                updated_value=get_state_value(car_a, car_b, pi_matrix, state_value_function)
                delta=max(abs(state_value_function[car_a][car_b]-updated_value), delta)
                state_value_function[car_a][car_b]=updated_value
        
        if(delta<theta):
            break
        
        print('delta: ', delta)
            
    # policy improvement
    policy_stable=True
    for car_a in range(num_cars_A):
        for car_b in range(num_cars_B):
            
            all_action_values=[]
            
            for action in possible_actions:
                current_val=get_state_value(car_a, car_b, pi_matrix, state_value_function)
                all_action_values.append(current_val)
                
            best_action=np.argmax(np.asarray(all_action_values))
            if(best_action==pi_matrix[car_a][car_b]):
                pass
            else:
                policy_stable=False
                pi_matrix[car_a][car_b]=best_action
    
    
    
    print(pi_matrix)
    plt.pcolor(pi_matrix)
    plt.show()
    if(policy_stable==True):
        break
                

plt.pcolor(pi_matrix)
plt.show()


# In[ ]:


plt.pcolor(pi_matrix)
plt.show()

#Reference from: https://stackoverflow.com/questions/11766536/matplotlib-3d-surface-from-a-rectangular-array-of-heights
from mpl_toolkits.mplot3d import Axes3D

X=np.arange(0,21)
Y=np.arange(0,21)
X,Y=np.meshgrid(X,Y)
fig=plt.figure()
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,state_value_function,rstride=1,cstride=1,cmap='hot',linewidth=0,antialiased=False)
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.show()


# # Question 7: Jack's Car Rental part 2

# In[ ]:


# Jacks' Car rental, ex. 4.7

import sys
from copy import deepcopy
import math
import matplotlib.pyplot as plt

# initialise parameters of MDP
num_cars_A=21
num_cars_B=21
num_states=num_cars_A*num_cars_B
grid_world=np.zeros(num_states)
num_actions=11
theta=1

# expectations of poisson distributions for the requests and returns for each location 
expected_requests_A=3
expected_requests_B=4
expected_returns_A=3
expected_returns_B=2

# max number of rentals or returns in a single day possible
max_number_cars_rented_or_returned=11
# discount rate
gamma=0.9

# rewards as specified
rent_reward=10
moving_reward=-2

# poisson distribution value for a given lambda and n
def poisson(n, lambda_val):
    return np.exp(-1*lambda_val)*np.power(lambda_val, n)/np.math.factorial(n)

# method to return expected return given a state and a chosen action
def get_state_value(car_a, car_b, pi_matrix, state_value_function):
    action=pi_matrix[car_a][car_b]
    moved_cars_a=car_a-action
    
    # additional constraint specified in question 4.7:
    # if we go from A to B, then number of cars transferred which is considered in reward becomes one less
    # else if we go from B to A, number of cars transferred which is considered in reward is same
    if(pi_matrix[car_a][car_b]>0):
        expected_return=moving_reward*abs(pi_matrix[car_a][car_b]-1)
    else:
        expected_return=moving_reward*abs(pi_matrix[car_a][car_b])
    
    moved_cars_b=car_b+action
    cars_in_a=max(0, min(moved_cars_a, num_cars_A-1))
    cars_in_b=max(0, min(moved_cars_b, num_cars_B-1))
    
    for request_sample_a in range(max_number_cars_rented_or_returned):
        for request_sample_b in range(max_number_cars_rented_or_returned):
            for return_sample_a in range(max_number_cars_rented_or_returned):
                for return_sample_b in range(max_number_cars_rented_or_returned):
                    final_prob=poisson(request_sample_a, expected_requests_A)
                    final_prob*=poisson(request_sample_b, expected_requests_B)
                    final_prob*=poisson(return_sample_a, expected_returns_A)
                    final_prob*=poisson(return_sample_b, expected_returns_B)
                    
                    rentals_A=min(cars_in_a, request_sample_a)
                    rentals_B=min(cars_in_b, request_sample_b)
                    
                    cars_in_a_end=int(min(cars_in_a+return_sample_a-rentals_A, num_cars_A-1))
                    cars_in_b_end=int(min(cars_in_b+return_sample_b-rentals_B, num_cars_B-1))
                    
                    reward=rent_reward*(rentals_A+rentals_B)
                    
                    # additional constraint mentioned in question:
                    # if more than 10 cars present in A or B, additional cost of 10 incurred
                    
                    if(cars_in_a_end>10):
                        reward-=4
                        
                    if(cars_in_b_end>10):
                        reward-=4
                    
                    expected_return+=final_prob*(reward+gamma*state_value_function[cars_in_a_end][cars_in_b_end])

    return expected_return


# policy matrix containing optimal actions for each state
pi_matrix=np.zeros((num_cars_A, num_cars_B))
# state value function for each state
state_value_function=np.zeros((num_cars_A, num_cars_B))

possible_actions=np.zeros(num_actions)
start_action=-5
for possible_action in range(num_actions):
    possible_actions[possible_action]=start_action
    start_action+=1


# Policy iteration    
while(True):
    
    # policy evaluation
    while(True):
        delta=0
        
        for car_a in range(num_cars_A):
            for car_b in range(num_cars_B):
                updated_value=get_state_value(car_a, car_b, pi_matrix, state_value_function)
                delta=max(abs(state_value_function[car_a][car_b]-updated_value), delta)
                state_value_function[car_a][car_b]=updated_value
        
        if(delta<theta):
            break
        
        print('delta: ', delta)
#         print(state_value_function)
            
    # policy improvement
    policy_stable=True
    for car_a in range(num_cars_A):
        for car_b in range(num_cars_B):
            
            all_action_values=[]
            
            for action in possible_actions:
                current_val=get_state_value(car_a, car_b, pi_matrix, state_value_function)
                all_action_values.append(current_val)
                
            best_action=np.argmax(np.asarray(all_action_values))
            if(best_action==pi_matrix[car_a][car_b]):
                pass
            else:
                policy_stable=False
                pi_matrix[car_a][car_b]=best_action
    
    
    
    print(pi_matrix)
    plt.pcolor(pi_matrix)
    plt.show()
    if(policy_stable==True):
        break
                

plt.pcolor(pi_matrix)
plt.show()


# In[ ]:


plt.pcolor(pi_matrix)
plt.show()

#Reference from: https://stackoverflow.com/questions/11766536/matplotlib-3d-surface-from-a-rectangular-array-of-heights
from mpl_toolkits.mplot3d import Axes3D

X=np.arange(0,21)
Y=np.arange(0,21)
X,Y=np.meshgrid(X,Y)
fig=plt.figure()
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,state_value_function,rstride=1,cstride=1,cmap='hot',linewidth=0,antialiased=False)
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.show()

