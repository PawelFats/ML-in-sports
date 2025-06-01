from ..plotters.base_plotter import BasePlotter
import matplotlib.pyplot as plt

class PlayerPlotter(BasePlotter):
    def _prepare(self, df, season_id=None):
        # Можно отфильтровать df по season_id, если нужно
        return df

    def _draw(self, df, season_id=None):
        figs = []
        # Пример: столбчатая диаграмма рейтинга
        fig1, ax1 = plt.subplots()
        df.sort_values('общий_рейтинг', ascending=False).plot(
            x='ID игрока', y='общий_рейтинг', kind='bar', ax=ax1)
        figs.append(fig1)

        # Пример: линия количества игр
        fig2, ax2 = plt.subplots()
        df.groupby('ID игрока')['игры'].sum().plot(ax=ax2)
        figs.append(fig2)

        return figs