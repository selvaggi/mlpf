import numpy as np
from scipy.optimize import curve_fit
from math import exp

def gaus(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2 / (2*sigma**2))


def fit_gaussian_iterative(data, bins=700, max_iter=10, tol=1e-4,
                           window_factor_low=3.0, window_factor_high=3.0):
    """
    Iteratively fits Gaussian inside range [mu - σ, mu + 2σ] until converged.

    window_factor_low  = 1 → fit region start = mu - 1σ
    window_factor_high = 2 → fit region end   = mu + 2σ
    """

    # Initial histogram + first crude estimates
    hist, edges = np.histogram(data, bins=bins, density=True)
    centers = (edges[1:] + edges[:-1]) / 2

    mu = np.sum(hist * centers) / np.sum(hist)                      # weighted mean
    sigma = np.sqrt(np.sum(hist * (centers - mu)**2) / np.sum(hist))# weighted std
    # print("mu, sigma", mu, sigma)
    for _ in range(max_iter):

        # restrict to range [mu-σ, mu+2σ]
        mask = (centers >= mu - window_factor_low*sigma) & (centers <= mu + window_factor_high*sigma)
        x_fit = centers[mask]
        y_fit = hist[mask]

        # fit
        try:
            popt, pcov = curve_fit(gaus, x_fit, y_fit, p0=[max(y_fit), mu, sigma])
        except:
            print("fit failed — using last values")
            return mu, sigma, None

        A_new, mu_new, sigma_new = popt
        

        # check convergence
        if abs(mu_new-mu) < tol and abs(sigma_new-sigma) < tol:
            mu, sigma = mu_new, sigma_new
            break
        # print("mu_new, sigma_new", mu_new, sigma_new)
        mu, sigma = mu_new, sigma_new

    return mu, sigma, pcov if 'pcov' in locals() else None


def plot_gaussian_with_window(data, mu, sigma, window_factor_low=1.0, window_factor_high=2.0, ax=None, color="red"):
    """
    Plots histogram and Gaussian between [mu−σ, mu+2σ].
    """
    # Histogram


    # Gaussian curve (full range)
    x = np.linspace(mu - window_factor_low * sigma, mu + window_factor_high * sigma, 1000)
    A = 1/(sigma*np.sqrt(2*np.pi))
    gaussian = A * np.exp(-(x-mu)**2/(2*sigma**2))

    # Window boundaries
    x_low = mu - window_factor_low * sigma
    x_high = mu + window_factor_high * sigma

    # # Mask for region [μ−σ , μ+2σ]
    # mask = (x >= x_low) & (x <= x_high)

    # Full Gaussian curve
    ax.plot(x, gaussian, linewidth=2, color=color)

    # # Highlight window region
    # plt.fill_between(x[mask], gaussian[mask], alpha=0.45,
    #                  label=f"Fit region [{x_low:.3f}, {x_high:.3f}]")

    # # Vertical lines for reference
    # ax.axvline(mu, color='k', linestyle='--', label="μ")
    # ax.axvline(x_low, color='r', linestyle='--', label="μ - σ")
    # ax.axvline(x_high, color='g', linestyle='--', label="μ + 2σ")
