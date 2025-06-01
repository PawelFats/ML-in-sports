from ..methods.base import IRatingStrategy

class StatsCalculator:
    def __init__(self, strategy: IRatingStrategy):
        self.strategy = strategy

    def compute(self, df):
        return self.strategy.calculate(df)