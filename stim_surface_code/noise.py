"""TODO
"""
from dataclasses import field
import copy
from typing_extensions import Self
import numpy as np

class NoiseParams:
    """Describes the noise affecting a quantum device.
    
    Attributes:
        baseline_error_means: Mean error vals (gate errors, T1/T2).
        baseline_error_stdevs: Standard deviations of error vals.
        error_distributions_log: Whether each error val is distributed on log
            scale.
    """
    baseline_error_means: dict[str, float]
    baseline_error_stdevs: dict[str, float]
    error_distributions_log: dict[str, bool]

    def __init__(
            self, 
            baseline_error_means: dict[str, float],
            baseline_error_stdevs: dict[str, float] = {
                'T1':0, 
                'T2':0, 
                'gate1_err':0, 
                'gate2_err':0, 
                'readout_err':0,
            },
            error_distributions_log: dict[str, bool] = {
                'T1':False, 
                'T2':False, 
                'gate1_err':True, 
                'gate2_err':True, 
                'readout_err':True,
            },
        ):
        """Initialize.

        Args:
            baseline_error_means: Mean error vals (gate errors, T1/T2).
            baseline_error_stdevs: Standard deviations of error vals.
            error_distributions_log: Whether each error val is distributed on log
                scale.
        """
        self.baseline_error_means = baseline_error_means
        self.baseline_error_stdevs = baseline_error_stdevs
        self.error_distributions_log = error_distributions_log

    def improve_noise_params(self, improvement_factor) -> Self:
        """Return a new NoiseParams where all values have been improved by some
        multiplicative factor. Does NOT modify itself.

        Args:
            improvement_factor: Factor to improve by.
        """
        new_noise_params = copy.deepcopy(self)
        for k,v in new_noise_params.baseline_error_means.items():
            if k == 'T1' or k == 'T2':
                new_noise_params.baseline_error_means[k] = v * improvement_factor
            else:
                new_noise_params.baseline_error_means[k] = v - np.log10(improvement_factor)
        return new_noise_params

StandardIdenticalNoiseParams = NoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':-5, 
        'gate2_err':-4, 
        'readout_err':-4
    },
)

GoogleIdenticalNoiseParams = NoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':-4, # to give average gate error around 0.08%
        'gate2_err':-2.5, # to give average gate error around 0.5%
        'readout_err':-1.7
    },
)

GoogleNoiseParams = NoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':-4, # to give average gate error around 0.08%
        'gate2_err':-2.5, # to give average gate error around 0.5%
        'readout_err':-1.7 # to give average around 2%
    },
    baseline_error_stdevs= {
        'T1':2e-6,
        'T2':5e-6,
        'gate1_err':0.1,
        'gate2_err':0.1,
        'readout_err':0.1,
    }
)