"""Four independent fit sections; historical names are translated only here."""

from copy import deepcopy


MODELS = ("spe_response", "gaussian_peaks", "emg")


def resolve_fit_config(config):
    model = dict(config.get("model", {}))
    name = model.get("name")
    if name not in MODELS:
        raise ValueError("fit.model.name must be one of {}".format(MODELS))
    response = name == "spe_response"
    defaults = {
        "model": dict(name=name, **({"background": True} if name == "emg" else
                                   {"npe": 3, "backscatter": True})),
        "constraints": dict(prefit_bounds=True, bounds={}, fixed={}, priors={}),
        "statistic": dict(name="chi2" if response else "unbinned_nll", nbins=300),
        "optimizer": dict(name="iminuit", initialization="moments" if name == "emg" else
                          ("pulse_height" if response else "spectrum"),
                          strategy=2 if response else 1, tolerance=.01 if response else .1,
                          max_calls=20000, multistart=response, prefit_nbins=300,
                          yield_step=None if response else 1.0),
    }
    if name != "emg":
        defaults["constraints"]["weights"] = "poisson" if response else "free"
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError("Unknown fit sections: {}".format(sorted(unknown)))
    result = deepcopy(defaults)
    for section, values in config.items():
        unknown = set(values) - set(defaults[section])
        if unknown:
            raise ValueError("Unknown fit.{} options: {}".format(section, sorted(unknown)))
        result[section].update(deepcopy(dict(values)))
    if result["statistic"]["name"] not in ("chi2", "unbinned_nll"):
        raise ValueError("fit.statistic.name must be chi2 or unbinned_nll")
    if int(result["statistic"]["nbins"]) < 20:
        raise ValueError("fit.statistic.nbins must be at least 20")
    if int(result["optimizer"]["prefit_nbins"]) < 20:
        raise ValueError("fit.optimizer.prefit_nbins must be at least 20")
    if result["optimizer"]["name"] != "iminuit":
        raise ValueError("Only fit.optimizer.name=iminuit is implemented")
    initializers = ("moments",) if name == "emg" else ("pulse_height", "spectrum")
    if result["optimizer"]["initialization"] not in initializers:
        raise ValueError("{} initialization must be one of {}".format(name, initializers))
    if int(result["optimizer"]["strategy"]) not in (0, 1, 2):
        raise ValueError("fit.optimizer.strategy must be 0, 1 or 2")
    if result["optimizer"]["tolerance"] <= 0 or int(result["optimizer"]["max_calls"]) <= 0:
        raise ValueError("Optimizer tolerance and max_calls must be positive")
    if name != "emg":
        if int(result["model"]["npe"]) not in (1, 2, 3):
            raise ValueError("fit.model.npe must be 1, 2 or 3")
        if result["constraints"]["weights"] not in ("poisson", "free"):
            raise ValueError("fit.constraints.weights must be poisson or free")
    elif result["optimizer"]["multistart"]:
        raise ValueError("EMG does not have an occupancy multistart")
    return result


def charge_configuration(config, statistic, poisson, npe, backscatter,
                         mean_prior=None, sigma_prior=None):
    if config is not None:
        result = resolve_fit_config(config)
        if result["model"]["name"] == "emg":
            raise ValueError("ChargeSpectrumFitter requires a charge model")
        return result
    # Compatibility for saved YAML and Python callers. No algorithm uses this label.
    response = statistic == "kor_chi2"
    if statistic not in ("kor_chi2", "chi2", "unbinned_nll"):
        raise ValueError("Unknown charge_fit_statistic: " + str(statistic))
    if response and (mean_prior is not None or sigma_prior is not None):
        raise ValueError("Use fit.constraints.priors with response parameter names, not SPE Gaussian priors")
    priors = {}
    if mean_prior is not None:
        priors["mu_spe"] = list(mean_prior)
    if sigma_prior is not None:
        priors["sigma_spe"] = list(sigma_prior)
    return resolve_fit_config({
        "model": dict(name="spe_response" if response else "gaussian_peaks",
                      npe=npe, backscatter=backscatter),
        "constraints": dict(weights="poisson" if poisson else "free", priors=priors),
        "statistic": dict(name="chi2" if response else statistic),
    })


def timing_configuration(config):
    result = resolve_fit_config(config or {"model": {"name": "emg"}})
    if result["model"]["name"] != "emg":
        raise ValueError("TimingEMGFitter requires model.name=emg")
    return result
