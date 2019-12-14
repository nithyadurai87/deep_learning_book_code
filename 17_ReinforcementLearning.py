import gym
import numpy as np
import random

e = gym.make("Taxi-v3").env
e.render()

e.reset()
e.render()

print (e.action_space)
print (e.observation_space)

e.s = e.encode(3, 1, 2, 0) # (taxi row, taxi column, passenger index, destination index)
e.render()

print (e.s)
print (e.P[328])

epochs, penalties, reward = 0, 0, 0
while not reward == 20:
    action = e.action_space.sample()
    state, reward, done, info = e.step(action) # (284, -1, False, {'prob': 1.0})
    if reward == -10:
        penalties += 1     
    epochs += 1      
print("Timesteps taken:",epochs)
print("Penalties incurred:",penalties)

q_table = np.zeros([e.observation_space.n, e.action_space.n])
print (q_table[328])
total_epochs, total_penalties = 0, 0
for i in range(1, 100):
    state = e.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False        
    while not done:
        if random.uniform(0, 1) < 0.1:
            action = e.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
        next_state, reward, done, info = e.step(action)          
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])       
        new_value = (1 - 0.1) * old_value + 0.1 * (reward + 0.6 * next_max)
        q_table[state, action] = new_value
        if reward == -10:
            penalties += 1
        state = next_state
        epochs += 1 
    total_penalties += penalties
    total_epochs += epochs              

print (q_table[328])  
print("Timesteps taken:",epochs)
print("Penalties incurred:",penalties)
print("Average timesteps per episode:",(total_epochs / 100))
print("Average penalties per episode",(total_penalties / 100))

