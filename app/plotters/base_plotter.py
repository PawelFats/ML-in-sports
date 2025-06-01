from abc import ABC, abstractmethod

class BasePlotter(ABC):
    def plot(self, df, **kwargs):
        data = self._prepare(df)
        fig = self._draw(data, **kwargs)
        return self._style(fig)

    @abstractmethod
    def _prepare(self, df): pass

    @abstractmethod
    def _draw(self, data, **kwargs): pass

    def _style(self, fig):
        # общие настройки оформления
        return fig