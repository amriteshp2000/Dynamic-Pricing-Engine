# pricing_engine/evaluation/online/regret_tracking.py

class RegretTracker:
    def __init__(self):
        self.regret = []

    def update(self, optimal_reward, realized_reward):
        self.regret.append(optimal_reward - realized_reward)

    def cumulative(self):
        return sum(self.regret)
