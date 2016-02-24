
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]
import numpy as np

import colormaps as cmaps

import AnniesLasso as tc

def calculate_sparsity(model, tolerance=1e-6):

    is_nonzero = np.abs(model.theta) > tolerance

    N = len(model.vectorizer.label_names)
    sparsity_first_order_derivatives = np.mean(is_nonzero[:, 1:1+N])
    sparsity_second_order_derivatives = np.mean(is_nonzero[:, 1+N:])
    sparsity_total = np.mean(is_nonzero[:, 1:])

    return (sparsity_first_order_derivatives, sparsity_second_order_derivatives,
        sparsity_total)



def plot_sparsity(models, scale_factor, color="k", percent=True, label=None, 
    fig=None, **kwds):
    """
    Plot sparsity metrics as a function of regularization term.
    """

    if fig is None:
        fig, axes = plt.subplots(1, 3,
            sharex=True, sharey=True, figsize=(11.5, 3.5))
    else:
        axes = fig.axes

    S = np.array([calculate_sparsity(m) for m in models])

    N = len(models[0].vectorizer.label_names)
    M = len(models[0].vectorizer.terms)
    S_labels = (
        r"$\boldsymbol{\theta}_{1...%i}$" % N,
        r"$\boldsymbol{\theta}_{%i...%i}$" % (N+1, M),
        r"$\boldsymbol{\theta}_{1...%i}$" % M)

    Lambda = np.clip([m.regularization[0] for m in models], 1e-3, np.inf)
        
    #kwds["c"] = kwds["c"] * np.ones_like(Lambda)
    if percent:
        S *= 100.

    for i, ax in enumerate(axes):

        _ = np.argsort(Lambda)

        scat = ax.scatter(Lambda[_], S[_, i], c=color, **kwds)

        ax.plot(Lambda[_], S[_, i], c=color, lw=1.5, zorder=-1)

        # X
        ax.semilogx()
        ax.set_xlabel(r"$\Lambda$")

        # Y
        ax.semilogy()
        if ax.is_first_col():
            ax.set_ylabel(r"$\rm{Matrix}$ $\rm{density}$ $(\%)$")
        ax.set_title(S_labels[i])


        ax.axhline(100, c="k", lw=1.0, linestyle=":", zorder=-1)

    return fig




if __name__ == "__main__":

    from glob import glob

    scale_factors = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
    #plt.cm.register_cmap("viridis", cmaps.viridis)
    #cmap = #plt.cm.get_cmap(cmaps.inferno, len(scale_factors))
    import matplotlib.colors as cm
    cmap = cm.LinearSegmentedColormap.from_list("viridis", cmaps._viridis_data,
        len(scale_factors))


    fig = None
    for i, scale_factor in enumerate(scale_factors):

        filenames = glob("gridsearch-{0:.1f}-*.model".format(scale_factor))
        models = [tc.load_model(_) for _ in filenames]
        models = [model for model in models if model.is_trained]

        color = cmap(i)
        fig = plot_sparsity(models, scale_factor, fig=fig,
            color=color, s=50)
            
    # Joint things for prettyness.
    fig.axes[-1].set_ylim(0.1, 150)
    fig.axes[-1].set_xlim(10**(-0.5), 10**5.5)
    
    fig.tight_layout()

    _ = np.zeros_like(scale_factors)
    _ = fig.axes[-1].scatter(_, _, c=scale_factors, cmap=cmap,
        vmin=min(scale_factors), vmax=max(scale_factors))
    cax = fig.add_axes([0.9, fig.subplotpars.bottom, 0.02,
        fig.subplotpars.top - fig.subplotpars.bottom])
    cbar = plt.colorbar(_, cax=cax)
    ticks = np.linspace(min(scale_factors), max(scale_factors), 1 + len(scale_factors))
    ticks = ticks[:-1] + np.diff(ticks)/2.
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([r"${:.1f}$".format(_) for _ in scale_factors])
    cbar.ax.tick_params(width=0)
    cbar.set_label(r"$\rm{Scale}$ $\rm{factor,}$ $f$")
    fig.subplots_adjust(right=0.88)
    
    fig.savefig("papers/sparsity.pdf", dpi=300)
    

