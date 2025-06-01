from abc import ABC, abstractmethod
import pandas as pd

class IRatingStrategy(ABC):
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычислить рейтинг и вернуть DataFrame"""
        pass