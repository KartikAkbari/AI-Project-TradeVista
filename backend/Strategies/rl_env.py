import numpy as np

class TradingEnv:
    def __init__(self, returns, window_size=10, initial_cash=10000):
        self.returns = returns
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares_held = 0
        self.total_value = self.initial_cash
        return self._get_state()

    def _get_state(self):
        return self.returns[self.current_step - self.window_size : self.current_step]

    def step(self, action):
        reward = 0
        done = False
        current_return = self.returns[self.current_step]

        # === Actions: 0=Hold, 1=Buy, 2=Sell ===
        if action == 1:  # Buy
            if self.cash > 0:
                self.shares_held = self.cash / (1 + current_return)
                self.cash = 0
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.cash = self.shares_held * (1 + current_return)
                self.shares_held = 0

        self.total_value = self.cash + self.shares_held * (1 + current_return)
        reward = self.total_value - self.initial_cash

        self.current_step += 1
        if self.current_step >= len(self.returns):
            done = True

        next_state = self._get_state()
        return next_state, reward, done
