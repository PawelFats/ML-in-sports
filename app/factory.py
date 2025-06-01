from .methods.red_method import RedMethodStrategy
# from .strategies.integral_method import IntegralMethodStrategy
# from .strategies.bayesian_method import BayesianStrategy

class StrategyFactory:
    mapping = {
        'red': RedMethodStrategy,
        # 'integral': IntegralMethodStrategy,
        # 'bayesian': BayesianStrategy,
    }

    @staticmethod
    def get(name: str):
        if name in StrategyFactory.mapping:
            return StrategyFactory.mapping[name]()
        raise ValueError(f"Unknown strategy: {name}")