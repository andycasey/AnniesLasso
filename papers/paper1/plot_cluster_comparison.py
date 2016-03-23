"""
Plot some cluster abundance stuff compared to ASPCAP.
"""


import matplotlib
matplotlib.rcParams["text.usetex"] = True


import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from matplotlib.ticker import MaxNLocator


# x vs y
# membership criteria.

# C vs N

# O vs Na
# O vs Mg
# O vs Al
# O vs S

# C+N+O vs FE_H?
# Mg vs Al




def plot_cluster_comparison(data, cluster_name, membership, x_elements, 
    y_elements, used_cannon_for_target_selection=True, vel_lim=None,
    xlims=None, ylims=None):
    """
    membership should be same len as data
    """

    candidate_color, membership_color = ("#666666", "#3498DB")
    candidate_color, membership_color = ("#BBBBBB", "#3498DB")
    tc_suffix, aspcap_suffix = ("", "_ASPCAP")
    
    candidates = data["FIELD"] == cluster_name

    membership_kwds = {"s": 50, "lw": 1.5}
    candidate_kwds = {"s": 30, "marker": "+", "lw": 1.5}

    fig, axes = plt.subplots(6, 2, figsize=(5.1, 16))
    axes = np.array(axes).flatten()

    axes[0].set_visible(False)
    axes[1].set_visible(False)

    top_ax = plt.subplot(6, 1, 1)


    # Vhelio and FE_H_1 (our metallicity?)
    suffix = tc_suffix if used_cannon_for_target_selection else aspcap_suffix
    top_ax.scatter(
        data["VHELIO_AVG"][candidates], data["FE_H" + suffix][candidates], 
        facecolor=candidate_color, rasterized=True,
        label=r"$\texttt{{FIELD = {0}}}$".format(cluster_name),
        **candidate_kwds)
    top_ax.scatter(
        data["VHELIO_AVG"][membership], data["FE_H" + suffix][membership], 
        facecolor=membership_color, rasterized=True, **membership_kwds)
    top_ax.errorbar(
        data["VHELIO_AVG"][membership], data["FE_H" + suffix][membership],
        xerr=data["VERR"][membership], yerr=data["E_FE_H" + suffix][membership],
        rasterized=True, 
        fmt=None, ecolor="k", zorder=-1)


    N, M = len(data["VHELIO_AVG"][candidates]), len(data["VHELIO_AVG"][membership])
    top_ax.text(0.05, 0.95, r"${:,}$".format(N), color=candidate_color,
            verticalalignment="top", horizontalalignment="left",
            transform=top_ax.transAxes)  
    top_ax.text(0.05, 0.95 - 0.11, r"${:,}$".format(M), color=membership_color,
            verticalalignment="top", horizontalalignment="left",
            transform=top_ax.transAxes)  

    #top_ax.legend(frameon=True, fontsize=11, loc="upper left")

    top_ax.set_xlabel(r"$V_{\rm helio}$ $(\rm{km}$ $\rm{s}^{-1})$")
    if used_cannon_for_target_selection:
        top_ax.set_ylabel(r"$[\rm{Fe}/\rm{H}]$ $(\rm{The}$ $\rm{Cannon})$")
    else:
        top_ax.set_ylabel(r"$[\rm{Fe}/\rm{H}]$ $(\rm{ASPCAP})$")

    top_ax.set_title(r"$\rm{{{0}}}$ $\rm{{membership}}$ $\rm{{selection}}$".format(
        cluster_name))

    top_ax.xaxis.set_major_locator(MaxNLocator(4))
    top_ax.yaxis.set_major_locator(MaxNLocator(4))


    
    for j, (element_x, element_y) in enumerate(zip(x_elements, y_elements)):

        x_wrt_fe, y_wrt_fe = (True, True)

        if element_x.lower() == "fe":
            x_wrt_fe = False

        if element_y.lower() == "fe":
            y_wrt_fe = False

        # X/Y for The Cannon
        for i, (mask, color) \
        in enumerate(zip((candidates, membership), (candidate_color, membership_color))):

            xerr, yerr = None, None
            if "," in element_x:
                x = 0
                xerr = 0
                for each in element_x.split(","):
                    x += data["{0}_H{1}".format(each.upper(), tc_suffix)]
                    xerr += data["E_{0}_H{1}".format(each.upper(), tc_suffix)]**2

                    if x_wrt_fe:
                        x = x - data["FE_H{}".format(tc_suffix)]

                if x_wrt_fe:
                    xerr += data["E_FE_H{0}".format(tc_suffix)]**2
                xerr = np.sqrt(xerr)

            else:
                x = data["{0}_H{1}".format(element_x.upper(), tc_suffix)]
                if x_wrt_fe:
                    x = x - data["FE_H{}".format(tc_suffix)]
                    xerr = (
                        data["E_{0}_H{1}".format(element_x.upper(), tc_suffix)]**2 + \
                        data["E_FE_H{0}".format(tc_suffix)]**2)**0.5
                
                else:
                    xerr = data["E_{0}_H{1}".format(element_x.upper(), tc_suffix)]
                

            if "," in element_y:
                y = 0
                yerr = 0
                for each in element_y.split(","):
                    y += data["{0}_H{1}".format(each.upper(), tc_suffix)]
                    yerr += data["E_{0}_H{1}".format(each.upper(), tc_suffix)]**2

                    if y_wrt_fe:
                        y = y - data["FE_H{}".format(tc_suffix)]

                if y_wrt_fe:
                    yerr += data["E_FE_H{}".format(tc_suffix)]**2
                yerr = np.sqrt(yerr)

            else:
                y = data["{0}_H{1}".format(element_y.upper(), tc_suffix)]
                if y_wrt_fe:
                    y = y - data["FE_H{}".format(tc_suffix)]
                    yerr = (
                        data["E_{0}_H{1}".format(element_y.upper(), tc_suffix)]**2 + \
                        data["E_FE_H{0}".format(tc_suffix)]**2
                        )**0.5
                else:
                    yerr = data["E_{0}_H{1}".format(element_y.upper(), tc_suffix)]


            kwds = candidate_kwds if i == 0 else membership_kwds
            axes[2*j + 2 + 1].scatter(x[mask], y[mask], facecolor=color, rasterized=True, **kwds)
            if xerr is not None and yerr is not None and color == membership_color:
                axes[2*j + 2 + 1].errorbar(x[mask], y[mask],
                    xerr=xerr[mask], yerr=yerr[mask], 
                    fmt=None, ecolor="k", zorder=-1, rasterized=True)

            # Quote the number of points.
            axes[2*j + 2 + 1].text(0.05, 0.95 - i * 0.10, r"${:,}$".format(len(x[mask])),
                color=color,
                verticalalignment="top", horizontalalignment="left",
                transform=axes[2*j + 2 + 1].transAxes)


        if xlims is None:
            tc_xlims = axes[2*j + 2 + 1].get_xlim()
            percent = 0.20 # 10%
            half_ptp = (np.ptp(tc_xlims) * (1 + percent))/2.
            tc_xlims = (np.mean(tc_xlims) - half_ptp, half_ptp + np.mean(tc_xlims))

        else:
            tc_xlims = xlims

        if ylims is None:
            tc_ylims = axes[2*j + 2 + 1].get_ylim()
            # Expand the scale just a little bit.
            percent = 0.20 # 10%
            half_ptp = (np.ptp(tc_ylims) * (1 + percent))/2.
            tc_ylims = (np.mean(tc_ylims) - half_ptp, half_ptp + np.mean(tc_ylims))
        else:
            tc_ylims = ylims

        # X/Y for ASPCAP.
        for i, (mask, color) \
        in enumerate(zip((candidates, membership), (candidate_color, membership_color))):

            if "," in element_x:
                x = 0
                for each in element_x.split(","):
                    x += data["{0}_H{1}".format(each.upper(), aspcap_suffix)]
                    if x_wrt_fe:
                        x = x - data["FE_H{}".format(aspcap_suffix)]
            else:
                x = data["{0}_H{1}".format(element_x.upper(), aspcap_suffix)]
                if x_wrt_fe:
                    x = x - data["FE_H{}".format(aspcap_suffix)]


            if "," in element_y:
                y = 0
                for each in element_y.split(","):
                    y += data["{0}_H{1}".format(each.upper(), aspcap_suffix)]
                    if y_wrt_fe:
                        y = y - data["FE_H{}".format(aspcap_suffix)]
            else:
                y = data["{0}_H{1}".format(element_y.upper(), aspcap_suffix)]
                if y_wrt_fe:
                    y = y - data["FE_H{}".format(aspcap_suffix)]

            kwds = candidate_kwds if i == 0 else membership_kwds
            axes[2*j + 2].scatter(x[mask], y[mask], facecolor=color, rasterized=True, **kwds)

            N = sum((tc_xlims[1] > x[mask]) * (x[mask] > tc_xlims[0]) \
                  * (tc_ylims[1] > y[mask]) * (y[mask] > tc_ylims[0]))
            axes[2*j + 2].text(0.05, 0.95 - i * 0.10, r"${:,}$".format(N), color=color,
                verticalalignment="top", horizontalalignment="left",
                transform=axes[2*j + 2].transAxes)  

        
        if j == 0:
            axes[2*j + 2].set_title(r"${\rm ASPCAP}$", y=1.05)
            axes[2*j + 2 + 1].set_title(r"${\rm The}$ ${\rm Cannon}$", y=1.05)
    

        for ax in (axes[2*j + 2], axes[2*j + 2 + 1]):
            ax.set_xlim(tc_xlims)
            ax.set_ylim(tc_ylims)

            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))

            ax.set_xlabel(r"$[\rm{{{0}}}/\rm{{{1}}}]$".format(element_x.title(),
                "Fe" if x_wrt_fe else "H"))
        
        if "," in element_y:
            axes[2*j + 2].set_ylabel(r"$[(\rm{{{0}}})/{{{1}}}\rm{{{2}}}]$".format(
                element_y.replace(",", "+"), element_y.count(",") + 1,
                "Fe" if y_wrt_fe else "H"))
        else:
            axes[2*j + 2].set_ylabel(r"$[\rm{{{0}}}/\rm{{{1}}}]$".format(element_y.title(),
                    "Fe" if y_wrt_fe else "H"))
        axes[2*j + 2 + 1].yaxis.set_ticklabels([])

        
    for ax in axes[2:]:
        ax.set(adjustable='box-forced', aspect=np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

    fig.tight_layout()

    if vel_lim is not None:
        top_ax.set_xlim(vel_lim)

    fig.subplots_adjust(hspace=-0.0, bottom=0.03)
    pos = top_ax.get_position()
    top_ax.set_position([pos.x0, pos.y0 + 0.06, pos.width, pos.height - 0.06])

    return fig


if __name__ == "__main__":

    # To speed up development cycle..
    try:
        data
    except:

        catalog = Table.read("../tc-cse-regularized-apogee-catalog.fits.gz")
        ok = catalog["OK"] * (catalog["R_CHI_SQ"] < 3) * (catalog["TEFF"] > 4000) * (catalog["TEFF"] < 5500)
        data = catalog[ok]



    """
    M67 186
    N6791 173
    N2243 187
    N188 252
    N1333 380
    N6819 307
    N7789 327

    M35N2158 189
    M54SGRC1 386
    M5PAL5 317
    """


    # M 15
    M15_members = (data["FIELD"] == "M15") \
                * (data["VHELIO_AVG"] > -130) * (data["VHELIO_AVG"] < 80) \
                * (data["FE_H"] < -1.7)
    
    M15_figure = plot_cluster_comparison(data, "M15", M15_members, 
        ["C", "O", "Mg", "Ca", "Fe"], ["N", "Na", "Al", "S", "C,N,O"],
        vel_lim=(-400, 150))
    M15_figure.savefig("M15_comparison.pdf", dpi=300)


    # M13
    M13_members = (data["FIELD"] == "M13") \
                * (data["VHELIO_AVG"] > -265) * (data["VHELIO_AVG"] < -220) \
                * (data["FE_H"] < -1.2)
    M13_figure = plot_cluster_comparison(data, "M13", M13_members, 
        ["C", "O", "Mg", "Ca", "Fe"], ["N", "Na", "Al", "S", "C,N,O"],
        )
    M13_figure.savefig("M13_comparison.pdf", dpi=300)


    # M 92
    M92_members = (data["FIELD"] == "M92") \
                * (data["VHELIO_AVG"] > -140) * (data["VHELIO_AVG"] < -100) \
                * (data["FE_H"] < -1.7)
    M92_figure = plot_cluster_comparison(data, "M92", M92_members, 
        ["C", "O", "Mg", "Ca", "Fe"], ["N", "Na", "Al", "S", "C,N,O"],
        )
    M92_figure.savefig("M92_comparison.pdf", dpi=300)


    # M 53
    M53_members = (data["FIELD"] == "M53") \
                * (data["VHELIO_AVG"] > -80) * (data["VHELIO_AVG"] < -40) \
                * (data["FE_H"] < -1.5)
    M53_figure = plot_cluster_comparison(data, "M53", M53_members, 
        ["C", "O", "Mg", "Ca", "Fe"], ["N", "Na", "Al", "S", "C,N,O"],
        )
    M53_figure.savefig("M53_comparison.pdf", dpi=300)


        

    # M 3
    M3_members = (data["FIELD"] == "M3") \
               * (data["VHELIO_AVG"] > -165) * (data["VHELIO_AVG"] < -120) \
               * (data["FE_H"] > -1.65) * (data["FE_H"] < -1.2)

    M3_figure = plot_cluster_comparison(data, "M3", M3_members,
        ["C", "O", "Mg", "Ca", "Fe"], ["N", "Na", "Al", "S", "C,N,O"])
    M3_figure.savefig("M3_comparison.pdf", dpi=300)

