
import numpy as np
import matplotlib
#matplotlib.rcParams["text.usetex"] = True
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from . import L1RegularizedCannonModel
from . import model, regularized

def _get_pixel_mask(model, wavelengths=None, pixels=None):

    if pixels is None:
        pixels = []

    if wavelengths is not None:
        pixels.extend(np.searchsorted(model.dispersion, wavelengths))

    pixel_mask = np.zeros_like(model.dispersion, dtype=bool)
    for pixel in pixels:
        pixel_mask[pixel] = True

    return pixels, pixel_mask


def compare_derivatives(models, model_names, latex_labels=None, **kwargs):
    """
    Compare the model derivatives (theta coefficients) for multiple models.
    """

    if not all(m.is_trained for m in models):
        raise ValueError("not all models are trained")


    axes_per_page = 10.
    N_axes = 1.0 + len(models[0].vectorizer.terms) 
    N_figs = int(np.ceil(N_axes/axes_per_page))

    figs = []
    for i in range(N_figs):
        fig, _ = plt.subplots(min([axes_per_page, N_axes]))
        figs.append(fig)

    axes = np.hstack([fig.axes for fig in figs])
    colours = ("#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD")
    
    if latex_labels is not None:
        label_names = models[0].vectorizer.get_human_readable_label_vector(
            latex_labels, mul=kwargs.pop("mul", "\cdot"))
    else:
        label_names = models[0].vectorizer.get_human_readable_label_vector()


    for i, ax in enumerate(axes):
        
        if 1 + i == N_axes:
            # Scatter plot
            break

        for m, name in zip(models, model_names):
            ax.plot(m.dispersion, m.theta[:, i], c=c, lw=2, label=name)

        ax.set_xlim(m.dispersion[0], m.dispersion[-1])
        if ax.is_last_row():
            ax.set_xlabel("Lambda")

        else:
            ax.set_xticklabels([])

        ax.set_ylabel(r"$\theta_{{{0}}}$".format(i))
        ax_twin = ax.twinx()
        ax_twin.set_ylabel(label_names[i] if latex_labels is None \
            else r"${}$".format(label_names[i]),
            rotation=0, labelpad=40)
        ax_twin.set_yticks([])


    # s2
    for m, name in zip(models, model_names):
        ax.plot(m.dispersion, m.s2, c=c, lw=2, label=name)

    if show_legend:
        figs[0].legend(figs[0].axes[0].lines, model_names,
            bbox_to_anchor=(0.5, 1.00),
            loc="upper center", frameon=False, ncol=len(models))


    return figs





def Lambda_lambda(models):
    """
    Show the minimum regularization parameter Lambda for each pixel with
    wavelength lambda.

    :param models:
        A list of trained L1RegularizedCannonModel objects for comparison which
        have only been trained on a subset of the labelled set.
    """

    model = models[0]

    validate_set = \
        (model._metadata["q"] % model._metadata.get("mod", 5)) == 0

    validation_scalar = np.zeros((len(models), len(model.dispersion)))
    validation_flux = model.normalized_flux[validate_set]
    validation_ivar = model.normalized_ivar[validate_set]

    Lambdas = np.zeros_like(validation_scalar)
    for i, model in enumerate(models):
        
        Lambdas[i] = model.regularization
        
        inv_var = validation_ivar / (1. + validation_ivar * model.s2)
        validation_scalar[i, :] = \
            regularized._fit_pixel_with_fixed_regularization_and_fixed_scatter(
                model.theta, model.s2**0.5, validation_flux, validation_ivar,
                inv_var, axis=1)

        """
        # Predict the fluxes in the validate set.
        inv_var = validation_ivar / (1. + validation_ivar * model.s2)
        design_matrix = model.vectorizer(np.vstack(
            [model.labelled_set[label_name][validate_set] \
                for label_name in model.vectorizer.label_names]).T)

        # Calculate the validation scalar.
        validation_scalar[i, :] = \
            _chi_sq(model.theta, design_matrix, validation_flux.T, inv_var.T,
                axis=1) 
        """

    # Scale the validation scalar (even though we don't need to, really..)
    scaled_validation_scalar = validation_scalar - validation_scalar[0, :]
    scaled_validation_scalar = scaled_validation_scalar/validate_set.sum()


    def get_best_Lambda_index_by_min(v):
        return np.argmin(v, axis=0)

    def get_best_Lambda_index_by_tol(v, tol=0.1):
        return np.where(v < np.min(v) + tol)[0][-1]


    x = np.tile([model.dispersion], validation_scalar.shape[0]).reshape(Lambdas.shape)
    y = np.log10(Lambdas)
    z = scaled_validation_scalar

    fig, ax = plt.subplots(1, 2)
#    ax.scatter(
#        
#        np.log10(Lambdas).flatten(), c=scaled_validation_scalar)
    #ax.imshow(z, interpolation="nearest")

    l = Lambdas[:, 0]
    ax[0].scatter(model.dispersion,
        [np.log10(l[get_best_Lambda_index_by_min(validation_scalar[:, i])]) for i in range(model.dispersion.size)],
        facecolor="k")

    ax[0].scatter(model.dispersion,
        [np.log10(l[get_best_Lambda_index_by_tol(validation_scalar[:, i])]) for i in range(model.dispersion.size)],
        facecolor="r")

    for i in range(model.dispersion.size):
        ax[1].plot(np.log10(Lambdas[:, i]), scaled_validation_scalar[:, i], c='k')

    ax[1].set_ylim(scaled_validation_scalar.min() - 1, 2)

    Lambdas = Lambdas[:, 0] # TODO check they are all the same
    y_index = np.argmin(scaled_validation_scalar, axis=0)
    y = Lambdas[y_index]


    raise a

    ax.scatter(model.dispersion, np.log10(y), facecolor="k")

    ax.set_xlabel(r"$\lambda$") # I bet Tufte would have *hated* astronomers.
    ax.set_ylabel(r"$\Lambda$")

    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(8))

    ax.set_yticklabels([r"$10^{%.0f}$" % _ for _ in ax.get_yticks()])

    fig.tight_layout()

    return fig



def label_residuals(model):
    """
    Plot the difference between the expected and inferred labels for the stars
    in the labelled set.

    :param model:
        The model to show the labelled residuals for.
    """

    label_names = model.vectorizer.label_names
    N_labels = len(label_names)

    fitted_labels = model.fit_labelled_set()

    fig, axes = plt.subplots(N_labels)
    for i, (ax, label_name) in enumerate(zip(axes, label_names)):

        ax.scatter(model.labelled_set[label_name], fitted_labels[:, i],
            facecolor='k')
        lims = [
            [_[0] for _ in ax.get_xlim()],
            [_[1] for _ in ax.get_xlim()]
        ]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims, c="#666666", zorder=-1)

        ax.set_xlabel(label_name)
        ax.set_ylabel("{} (fitted)".format(label_name))

    fig.tight_layout()
    return fig



def pixel_regularization_effectiveness(models, pixels, latex_labels=None, same_limits=False,
    show_legend=True, **kwargs):
    """
    Visualize the effectiveness of the regularization on a single pixel.

    :param models:
        A list of trained L1RegularizedCannonModel objects for comparison.

    :param wavelengths:
        The wavelengths to show the pixel regularization at.

    :param latex_labels: [optional]
        Specify LaTeX for the label names in the corresponding models.

    :param show_legend: [optional]
        Toggle legend display.
    """

    if 2 > len(models):
        raise ValueError("must provide more than a single model for comparison")

    for model in models:
        if not isinstance(model, L1RegularizedCannonModel):
            raise TypeError("models should be a list of trained "
                            "L1RegularizedCannonModel objects")

    if latex_labels is not None:
        label_names = models[0].vectorizer.get_human_readable_label_vector(
            latex_labels, mul=kwargs.pop("mul", "\cdot"))
    else:
        label_names = models[0].vectorizer.get_human_readable_label_vector()

    # Make some checks that the models are the same, etc.
    # same dispersion, q values, etc.
    #pixels = np.searchsorted(models[0].dispersion, wavelengths)
    #pixels = np.array([0, 1])
    wavelengths = models[0].dispersion[pixels]

    # Sort the models by their regularization value.
    N = models[0].theta.shape[1]
    Lambdas = np.array([model.regularization for model in models])
    
    s2 = np.array([model.s2 for model in models])
    theta = np.array([model.theta for model in models])

    colours = ("#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD")
    xlims = np.log10([np.nanmin(Lambdas), np.nanmax(Lambdas)])

    if N > 10:
        M = int(np.ceil((N + 1)/10.))
        figs = []
        for m in range(M):
            fig, axes = plt.subplots(10, figsize=(8, 8))
            figs.append(fig)

    else:
        fig, axes = plt.subplots(N + 1, figsize=(8, 8))
        figs = [fig]

    axes = np.array([fig.axes for fig in figs]).flatten()
    for i, ax in enumerate(axes[:-1]):

        if i >= theta.shape[2]: break
        for j, (wavelength, pixel) in enumerate(zip(wavelengths, pixels)):
            
            x = np.log10(Lambdas[:, pixel])
            y = theta[:, pixel, i]

            _ = np.argsort(x)
            x, y = x[_], y[_]
            ax.plot(x, y, label=wavelength, lw=2, c=colours[j])

            # Do the fill between.
            if i != 0:
                ax.fill_between(x, y, facecolor=colours[j], alpha=0.25)

        if ax.is_first_row():
            ax.yaxis.set_major_locator(MaxNLocator(2))
            ylim = ax.get_ylim()
            ax.set_yticks([
                np.mean(ylim) - np.ptp(ylim)/4.,
                np.mean(ylim) + np.ptp(ylim)/4.
            ])

        else:
            ax.axhline(0, c='k', lw=1)

        # Ticks and labels, etc.
        ax.set_xlim(xlims)
        ax.set_xticklabels([])

        ax.xaxis.set_major_locator(MaxNLocator(4))
        
        ax.set_ylabel(r"$\theta_{{{0}}}$".format(i))

        ax_twin = ax.twinx()
        ax_twin.set_ylabel(label_names[i] if latex_labels is None \
            else r"${}$".format(label_names[i]),
            rotation=0, labelpad=40)
        ax_twin.set_yticks([])

    
    ax = axes[-1]
    lines = []
    for j, (wavelength, pixel) in enumerate(zip(wavelengths, pixels)):
        x = np.log10(Lambdas[:, pixel])
        y = s2[:, pixel]
        _ = np.argsort(x)
        x, y = x[_], y[_]
        lines.append(ax.plot(x, y, c=colours[j], lw=2,
            label=r"$%0.1f\,{\rm \AA}$" % (wavelength, )))
        ax.set_xlim(xlims)

    
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.locator_params(axis='y', nbins=3)
    ax.set_ylabel(r"$s^2$")

    if same_limits:
        ylim = np.max(np.abs([ax.get_ylim() for ax in axes[1:-1] if len(ax.lines) > 0]))
        for ax in axes[1:-1]:
            ax.set_ylim(-ylim, +ylim)
            ax.set_yticks([-ylim/2., +ylim/2.])

            ax.set_xlim(xlims)    
            ax.xaxis.set_major_locator(MaxNLocator(4))

            if ax.is_last_row():
                ax.set_xlabel(r"$\Lambda$")    
                ax.set_xticklabels([r"$10^{%.0f}$" % _ for _ in ax.get_xticks()])

    else:

        for ax in axes[1:-1]:
            ylim = np.max(np.abs(ax.get_ylim()))
            ax.set_ylim(-ylim, +ylim)
            ax.set_yticks([-ylim/2., +ylim/2.])

            ax.set_xlim(xlims)    
            ax.xaxis.set_major_locator(MaxNLocator(4))

            if ax.is_last_row():
                ax.set_xlabel(r"$\Lambda$")    
                ax.set_xticklabels([r"$10^{%.0f}$" % _ for _ in ax.get_xticks()])

           
    if show_legend:
        fig.legend(fig.axes[0].lines, 
            [r"$%0.1f\,{\rm \AA}$" % _ for _ in wavelengths],
            bbox_to_anchor=(0.5, 1.00),
            loc="upper center", frameon=False, ncol=len(wavelengths))

    for fig in figs:        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.05, hspace=0.05, right=0.86, top=0.95)

    return figs


def calculate_optimal_lambda(Lambda, Q, tolerance=0.1, full_output=False):
    # Return just the minimum:
    #return Lambda[np.argmin(Q)]
    #if full_output:
    #    return (Lambda[np.argmin(Q)], np.argmin(Q))

    # Return max({Lambda: Q(Lambda) < Q_min + tolerance })
    index = np.where(Q < np.min(Q) + tolerance)[0][-1]
    if full_output:
        return (Lambda[index], index)
    return Lambda[index]




def pixel_regularization_validation(models, pixels, show_legend=True):
    """
    Show the balance between model prediction and regularization for given
    pixels.
    """

    wavelengths = models[0].dispersion[pixels]

    # Do the prediction for some spectra.

    N_px, N_models = (models[0].theta.shape[0], len(models))

    validation_scalar = np.zeros((N_models, N_px))
    validate_set = \
        (models[0]._metadata["q"] % models[0]._metadata.get("mod", 5)) == 0

    validation_ivar = models[0].normalized_ivar[validate_set]
    validation_flux = models[0].normalized_flux[validate_set]


    for i, m in enumerate(models):
        
        m.s2[~np.isfinite(m.s2)] = 0.0
        design_matrix = m.vectorizer(np.array([m.labelled_set[name][validate_set] for name in m.vectorizer.label_names]).T)
        inv_var = validation_ivar / (1. + validation_ivar * m.s2)
        validation_scalar[i, :] = \
            model._chi_sq(m.theta, design_matrix, validation_flux.T, inv_var.T,
                axis=1)

        #if not np.all(np.isfinite(validation_scalar[i, :])):
        #    raise a

        """
        # Predict the fluxes in the validate set.
        inv_var = m.normalized_ivar[validate_set] / \
            (1. + m.normalized_ivar[validate_set] * m.s2)
        design_matrix = m.vectorizer(np.vstack(
            [m.labelled_set[label_name][validate_set] \
                for label_name in m.vectorizer.label_names]).T)

        # Calculate the validation scalar.
        validation_scalar[i, :] = model._chi_sq(m.theta, design_matrix, 
            m.normalized_flux[validate_set].T, inv_var.T, axis=1)
        #validation_scalar[i, :] += model._log_det(inv_var)
        """

    #if not np.all(np.isfinite(validation_scalar[0, :])):
    #    raise a
    # Get regularization parameters.
    Lambda = np.array([m.regularization[0] for m in models])
    scaled_validation_scalar = validation_scalar - validation_scalar[0, :]
    #scaled_validation_scalar = scaled_validation_scalar/validate_set.sum()

    fig, ax = plt.subplots()
    ax.axhline(0, c='k', zorder=-1)
    colours = \
        ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD"]
    for pixel, wavelength in enumerate(wavelengths):

        xi = np.log10(Lambda)
        yi = scaled_validation_scalar[:, pixel]

        _ = np.argsort(xi)
        xi, yi = xi[_], yi[_]
        ax.plot(xi, yi, lw=2, c=colours[pixel % len(colours)],
            label=r"$%0.1f\,{\rm \AA}$" % wavelength)

        l, j = calculate_optimal_lambda(Lambda, yi, full_output=True)
        ax.scatter([np.log10(l)], [yi[j]], facecolor=colours[pixel % len(colours)],
            linewidths=2, edgecolor="k", s=100, zorder=100)


    if show_legend:
        ax.legend(loc="lower left", frameon=False)

    ax.set_xlabel(r"$\Lambda$")
    ax.set_xlim(np.log10(np.min(Lambda)), np.log10(np.max(Lambda)))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xticklabels([r"$10^{%.0f}$" % _ for _ in ax.get_xticks()])

    ymin = scaled_validation_scalar.min()
    ax.set_ylabel(r"$\chi^2 + \Delta$")
    ax.set_ylim(ymin - 1, 2)
    ax.yaxis.set_major_locator(MaxNLocator(6))

    fig.tight_layout()

    return fig









def regularization_validation(model, wavelengths=None, pixels=None,
    regularizations=None):

    # Get all the points in just pixel space.
    pixels, pixel_mask = _get_pixel_mask(model, wavelengths, pixels)

    if regularizations is None:
        regularizations = np.arange(0, 7, 0.05)

    # Do this for many regularization points
    # Plot all of the lines of chi^2 as a function of the regularization param.

    if len(pixels) > 1:
        raise NotImplementedError
    # Split the training set into 10 different components.
    components = np.random.randint(0, 10, size=len(model.labelled_set))

    train_component = components > 0
    validate_component = components == 0

    N_pixels = len(pixels)
    N_regularizations = len(regularizations)
    N_training_stars = train_component.sum()
    N_validation_stars = validate_component.sum()
    test_scalar = np.zeros((N_regularizations, N_validation_stars, N_pixels))
    for i, regularization in enumerate(regularizations):

        # Ignore the first component and train on the rest.
        new_model = L1RegularizedCannonModel(
            model.labelled_set[train_component],
            model.normalized_flux[train_component, pixel_mask].reshape(-1, N_pixels),
            model.normalized_ivar[train_component, pixel_mask].reshape(-1, N_pixels),
            model.dispersion[pixel_mask])
        new_model.vectorizer = model.vectorizer
        new_model.s2 = 0
        new_model.regularization = 10**regularization
        new_model.train(True)

        # Now predict the fluxes for the missing component.
        design_matrix = new_model.vectorizer(np.vstack(
            [model.labelled_set[label_name][validate_component] \
                for label_name in new_model.vectorizer.label_names]).T)

        norm_ivar = model.normalized_ivar[validate_component, pixel_mask]
        #inv_var = norm_ivar/(1. + norm_ivar * new_model.scatter**2)
        inv_var = norm_ivar

        residuals = np.dot(new_model.theta, design_matrix.T) \
            - model.normalized_flux[validate_component, pixel_mask].reshape(N_pixels, -1)

        test_scalar[i] = np.sum(inv_var * residuals**2, axis=0).reshape(-1, 1) #\
            #- np.sum(np.log(inv_var))


    fig, ax = plt.subplots()
    y = np.sum(test_scalar, axis=1).flatten()
    y -= y[0]
    ax.plot(regularizations, y, c="k", lw=2)
    ax.axhline(np.sum(test_scalar[0]), c="#666666", zorder=-1)

    # X-axis stuff.
    ax.set_xlim(regularizations[0], regularizations[-1])
    ax.set_ylim(y.min() - 1, 2)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel(r"$\Lambda$")
    ax.set_xticklabels([r"$10^{%.0f}$" % _ for _ in ax.get_xticks()])

    # Y-axis stuff (at high regularization this blows up..)
    ax.set_ylabel(r"$\sum\chi^2_{validate} + \Delta$")
    #ylim = min([ax.get_ylim()[1], np.sum(test_scalar[0]) * 10])
    #ax.set_ylim(0, ylim)
    #ax.yaxis.set_major_locator(MaxNLocator(6))
    #ax.set_yticklabels([r"${0:.0f}$".format(_) for _ in ax.get_yticks()])

    # Quote the number of stars in the training set, the validation set, and
    # the wavelength(s).
    ax.set_title(r"$\lambda\,=\,%0.3f\,{\rm \AA}$" % (model.dispersion[pixels[0]], ))
    ax.text(0.05, 0.90, r"$N_{train} = %.0f$" % N_training_stars,
        transform=ax.transAxes, horizontalalignment="left")
    ax.text(0.05, 0.85, r"$N_{validate} = %.0f$" % N_validation_stars,
        transform=ax.transAxes, horizontalalignment="left")
    
    return fig

