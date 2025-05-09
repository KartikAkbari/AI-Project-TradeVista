import pandas as pd
import numpy as np
from Strategies.rl_env import TradingEnv
from Strategies.dqn_agent import DQNAgent

df = pd.read_csv("Data/Processed/AAKASH.csv")
df['Return'] = df['close'].pct_change().dropna()
returns = df['Return'].dropna().values

env = TradingEnv(returns)
agent = DQNAgent(state_size=10, action_size=3)

episodes = 50
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay(batch_size=32)
        state = next_state
        total_reward += reward

    print(f"Episode {episode+1}/{episodes} | Total Reward: {total_reward:.2f}")
