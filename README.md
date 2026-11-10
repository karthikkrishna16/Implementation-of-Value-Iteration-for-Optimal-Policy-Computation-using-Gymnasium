# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement
Implement the Value Iteration algorithm on the FrozenLake environment to determine the optimal value of each state and derive the optimal policy that maximizes the expected cumulative reward. The algorithm should repeatedly update state values until convergence and then extract the best action for every state.




## Software Requirements
```
Python 
Gymnasium Library
NumPy
Jupyter Notebook/Google colab
```
## Environment Description

The environment is a 4×4 customized FrozenLake consisting of safe frozen cells (F), hole states (H), a start state (S), and a goal state (G). The agent begins at the Start (S) and must reach the Goal (G) while avoiding the holes. Since the environment uses is_slippery=True, the agent may not always move in the intended direction, making the environment stochastic. The Value Iteration algorithm is used to compute the optimal state-value function and the optimal policy for safely reaching the goal.
### Map Used:
```

F F H F
H S H F
H F F H
H F G H

```
```
Where:

S – Start State
F – Frozen Surface (Safe)
H – Hole (Terminal State)
G – Goal State (Reward = 1)

Since is_slippery=True, the agent may move in unintended directions due to slippery ice, making the environment stochastic. 
```
## MDP Representation

#### States (S)
There are 16 states in a 4×4 FrozenLake grid.

S={0,1,2,…,15}
Each grid cell represents one state. The agent starts from the Start (S) state and aims to reach the Goal (G) state while avoiding Hole (H) states.

#### Actions (A)

There are four possible actions.
```
Action	Value
Left	0
Down	1
Right	2
Up	3
```
### Transition Probability (P)
The transition probability defines the likelihood of moving from one state to another after taking an action.
Since is_slippery=True, the environment is stochastic, meaning the agent may move in an unintended direction. Therefore, the next state depends on both the selected action and the slipping probability.
``` 
P(s′∣s,a)
```

### Reward Function (R)
```
Goal State = +1
Hole = 0
Frozen State = 0
Discount Factor (γ)
γ=0.99
```
The discount factor gives high importance to future rewards.

## Theory
Value Iteration is a dynamic programming algorithm used to solve Markov Decision Processes (MDPs) by repeatedly applying the Bellman Optimality Equation until the value function converges. In each iteration, the algorithm updates the value of every state by selecting the action that provides the maximum expected future reward. Once the values converge, the optimal policy is obtained by choosing the action with the highest value for each state.
The Bellman Optimality Equation is:

```
V(s) = max_a Σ P(s' | s, a) [R(s, a, s') + γV(s')]
```

## Algorithm

1. Create the customized FrozenLake-v1 environment using the modified map.
2. Initialize the value of all states to zero.
3. Set the discount factor (γ = 0.99) and convergence threshold (θ = 1e−8).
4. Compute the expected value for every possible action in each state using the Bellman Optimality Equation.
5. Update each state's value with the maximum expected action value.
6. Repeat the updates until the change in state values is less than the convergence threshold.
7. Extract the optimal policy by selecting the action with the highest expected value for each state.
8. Display the number of iterations, optimal state-value function, and optimal policy.



## Python Program

```python
import gymnasium as gym
import numpy as np

# Action visualization dictionary
action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs value iteration, computes the optimal value function,
    and directly derives the optimal policy.
    """
    nS = env.observation_space.n
    nA = env.action_space.n

    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    iteration = 0

    while True:
        delta = 0
        for s in range(nS):
            action_values = np.zeros(nA)
            for a in range(nA):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            best_action_value = np.max(action_values)
            delta = max(delta, abs(best_action_value - V[s]))
            V[s] = best_action_value
            policy[s] = np.argmax(action_values)

        iteration += 1
        if delta < theta:
            break

    return V, policy, iteration

# =================================================
# PART 1: BEFORE CHANGING (Standard Configuration)
# =================================================
print("\n--- PART 1: BEFORE CHANGING (Standard Layout) ---")
print("Name: Anto Williams")
print("Register Number: 212224240012")
print("-" * 50)

standard_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env_std = gym.make("FrozenLake-v1", desc=standard_desc, is_slippery=True)
V_std, policy_std, iterations_std = value_iteration(env_std)

print("Value Iteration Completed")
print("Number of Iterations:", iterations_std)
print("\nOptimal State-Value Function (Before Change):")
print(np.round(V_std.reshape(4, 4), 4))

policy_grid_std = np.array(
    [action_symbols[action] for action in policy_std]
).reshape(4, 4)
print("\nOptimal Policy (Before Change):")
print(policy_grid_std)
env_std.close()

# =================================================
# PART 2: AFTER CHANGING (Customized Layout)
# =================================================
print("\n--- PART 2: AFTER CHANGING (Customized Layout) ---")
print("Name: Anto Williams")
print("Register Number: 212224240012")
print("-" * 50)

env_desc = [
    "FFHF",
    "HSHF",
    "HFFH",
    "HFGH"
]

env_custom = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
V_cust, policy_cust, iterations_cust = value_iteration(env_custom)

print("Value Iteration Completed")
print("Number of Iterations:", iterations_cust)
print("\nOptimal State-Value Function (After Change):")
print(np.round(V_cust.reshape(4, 4), 4))

policy_grid_cust = np.array(
    [action_symbols[action] for action in policy_cust]
).reshape(4, 4)
print("\nOptimal Policy (After Change):")
print(policy_grid_cust)
env_custom.close()







```

---

## Output


### Number of Iterations:

#### For Normal description:

<img width="565" height="120" alt="image" src="https://github.com/user-attachments/assets/3d41ded6-7cb0-4662-8735-a7c0e7d36840" />








#### After changing the description:
<img width="521" height="120" alt="image" src="https://github.com/user-attachments/assets/b52387d3-cfc2-488d-9503-5fea8dff607e" />











### Optimal State-Value Function:

#### For Normal description:

<img width="557" height="131" alt="image" src="https://github.com/user-attachments/assets/655dea27-d014-4f00-b99c-bf2f1e77ed52" />



#### After changing the description: 
<img width="486" height="125" alt="image" src="https://github.com/user-attachments/assets/8b94ae57-67ea-4729-b00e-5ea21115608b" />







### Optimal Policy:

#### For Normal description:
<img width="383" height="127" alt="image" src="https://github.com/user-attachments/assets/16458df8-637c-4a46-b846-56a07400c9a9" />



#### After changing description:
<img width="407" height="122" alt="image" src="https://github.com/user-attachments/assets/08877afe-d9b2-4d3d-8fbe-f506b35236b1" />






## Result

The Value Iteration algorithm was successfully implemented on the customized FrozenLake-v1 environment. It converged after several iterations and computed the optimal state-value function and optimal policy for the modified map. The resulting policy guides the agent from the start state to the goal while avoiding hole states as much as possible.

## Inference
The comparison between the original and modified FrozenLake environments shows that changing the map layout directly affects the Markov Decision Process. In the modified environment, the start state, goal state, and hole positions were rearranged, creating different state transitions and fewer safe paths. As a result, the optimal state-value function, policy, and convergence behavior differed from the original environment. The Value Iteration algorithm successfully adapted to these changes by recalculating the optimal values and selecting the best action for each state. This confirms that Value Iteration is robust and capable of finding the optimal policy for different MDP configurations.
