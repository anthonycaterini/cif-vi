import torch
import torch.utils.data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


_DTYPE = torch.get_default_dtype()


# TODO: Make return a matplotlib figure instead. Writing can be done outside.
class DensityVisualizer:
    def __init__(self, writer):
        self._writer = writer

    def visualize(self, density, epoch):
        raise NotImplementedError


class DummyDensityVisualizer(DensityVisualizer):
    def visualize(self, density, epoch):
        return


class TwoDimensionalVIVisualizer(DensityVisualizer):
    _NUM_POSTERIOR_SAMPLES = 5000
    _N_CONTOUR_LEVELS = 20
    _COLORMAP = "coolwarm"

    def __init__(self, writer, fig_x_limits, fig_y_limits, title=None):
        super().__init__(writer=writer)
        self.dummy_data = torch.zeros((self._NUM_POSTERIOR_SAMPLES, 1), dtype=_DTYPE)
        self.fig_x_limits = fig_x_limits
        self.fig_y_limits = fig_y_limits
        self.title = title

    def visualize(self, density, epoch):
        with torch.no_grad():
            samples = density.elbo(self.dummy_data)["approx-posterior-sample"]

        samples = samples.cpu().numpy()

        plt.figure()
        plt.scatter(samples[:,0], samples[:,1])
        self._writer.write_figure("approx-post-samples", plt.gcf(), epoch)

        plt.figure()
        plt.xlim(*self.fig_x_limits)
        plt.ylim(*self.fig_y_limits)

        # HACK: MAKE SEABORN PLOT FILL SPACE
        plt.gca().patch.set_color(cm.get_cmap(self._COLORMAP)(1./self._N_CONTOUR_LEVELS))

        sns.kdeplot(
            samples[:,0],
            samples[:,1],
            cmap=self._COLORMAP,
            n_levels=self._N_CONTOUR_LEVELS,
            shade=True,
            shade_lowest=False,
            cbar=True
        )

        if self.title: # XXX: For test visualization
            plt.title(self.title, fontsize=20)

        self._writer.write_figure("approx-post-density", plt.gcf(), epoch)
