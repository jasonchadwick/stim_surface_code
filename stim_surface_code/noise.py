"""TODO
"""
import copy
from typing_extensions import Self
import numpy as np
import qc_utils.stats
import scipy
from stim_surface_code.patch import SurfaceCodePatch

class NoiseParams:
    """Describes the noise affecting a quantum device.
    
    Attributes:
        baseline_error_means: Mean error vals (gate errors, T1/T2).
        baseline_error_stdevs: Standard deviations of error vals.
        distributions_log: Whether each error val is distributed on log
            scale.
    """
    error_means: dict[str, float]
    error_stdevs: dict[str, float]
    distributions_log: dict[str, bool]

    def __init__(
            self, 
            baseline_error_means: dict[str, float],
            baseline_error_stdevs: dict[str, float] = {
                'T1':0, 
                'T2':0, 
                'gate1_err':0, 
                'gate2_err':0, 
                'readout_err':0,
                'erasure':0,
            },
            distributions_log: dict[str, bool] = {
                'T1':False, 
                'T2':False, 
                'gate1_err':True, 
                'gate2_err':True, 
                'readout_err':True,
                'erasure':True,
            },
        ):
        """Initialize.

        Args:
            baseline_error_means: Mean error vals (gate errors, T1/T2).
            baseline_error_stdevs: Standard deviations of error vals.
            distributions_log: Whether each error val is distributed on log
                scale.
        """
        self.error_means = baseline_error_means
        self.error_stdevs = baseline_error_stdevs
        self.distributions_log = distributions_log

    def improve(self, improvement_factor) -> Self:
        """Return a new NoiseParams where all values have been improved by some
        multiplicative factor. Does NOT modify itself. Scales standard
        deviations as well.

        Args:
            improvement_factor: Factor to improve by.
        """
        new_noise_params = copy.deepcopy(self)
        for k in new_noise_params.error_means.keys():
            if k == 'T1' or k == 'T2':
                new_noise_params.error_means[k] *= improvement_factor
                new_noise_params.error_stdevs[k] *= improvement_factor
            else:
                new_noise_params.error_means[k] /= improvement_factor
                new_noise_params.error_stdevs[k] /= improvement_factor
        return new_noise_params
    
    def set_patch_err_vals(self, patch: SurfaceCodePatch):
        """Set the error values of the patch based on the noise params.
        
        Args:
            patch: The patch to set error values for.
        """
        maxvals = {
            'T1': np.inf,
            'T2': np.inf,
            'readout_err': 1.0,
            'gate1_err': 3/4,
            'gate2_err': 15/16,
            'erasure': 1.0,
        }
        minvals = {
            'T1': 0.0,
            'T2': 0.0,
            'readout_err': 0.0,
            'gate1_err': 0.0,
            'gate2_err': 0.0,
            'erasure': 0.0,
        }

        assert all([v >= minvals[k] for k,v in self.error_means.items()]), 'Mean values below minimum'
        assert all([v <= maxvals[k] for k,v in self.error_means.items()]), 'Mean values above maximum'
        assert all([v >= 0 for v in self.error_stdevs.values()]), 'Standard deviations must be nonnegative'

        error_val_dict_keys = {
            'T1': [q.idx for q in patch.all_qubits],
            'T2': [q.idx for q in patch.all_qubits],
            'readout_err': [q.idx for q in patch.all_qubits],
            'gate1_err': [q.idx for q in patch.all_qubits],
            'gate2_err': list(patch.qubit_pairs),
            'erasure': [q.idx for q in patch.all_qubits],
        }
        
        error_vals = {}
        for k,mean in self.error_means.items():
            if self.distributions_log[k]:
                if self.error_stdevs[k] == 0:
                    vals = np.full(len(error_val_dict_keys[k]), mean)
                else:
                    vals = np.clip(qc_utils.stats.lognormal(mean, self.error_stdevs[k], size=len(error_val_dict_keys[k])), minvals[k], maxvals[k])
            else:
                vals = np.clip(np.random.normal(mean, self.error_stdevs[k], size=len(error_val_dict_keys[k])), minvals[k], maxvals[k])
            error_vals[k] = {k:vals[i] for i,k in enumerate(error_val_dict_keys[k])}

        patch.set_error_vals(error_vals)

StandardIdenticalNoiseParams = NoiseParams(
    {
        'T1':200e-6, 
        'T2':300e-6, 
        'gate1_err':1e-5, 
        'gate2_err':1e-4, 
        'readout_err':1e-4,
        'erasure':0,
    },
)

def get_T1T2_gate_err(t1, t2, gate_time):
    p_x = max(0, 0.25 * (1 - np.exp(-gate_time / t1)))
    p_y = p_x
    p_z = max(0, 0.5 * (1 - np.exp(-gate_time / t2)) - p_x)
    return (p_x+p_y+p_z)

def get_T1T2_limited_params_by_err_rate(target_gate2_err_rate: float, gate2_time: float = 34e-9, t1_over_t2_ratio: float = 1.0):
    """TODO
    """
    t2 = scipy.optimize.minimize(lambda t2: abs(1/target_gate2_err_rate - 1/(2*get_T1T2_gate_err(t1_over_t2_ratio*t2, t2, gate2_time))), 100e-6).x[0]
    t1 = t1_over_t2_ratio * t2
    return (t1, t2)

GoogleNoiseParams = NoiseParams(
    {
        'T1':20e-6, 
        'T2':30e-6, 
        'gate1_err':1e-3 - get_T1T2_gate_err(20e-6, 30e-6, 34e-9),
        'gate2_err':5e-3 - 2*get_T1T2_gate_err(20e-6, 30e-6, 25e-9),
        'readout_err':2e-3,
        'erasure':0,
    },
    baseline_error_stdevs= {
        'T1':2e-6,
        'T2':5e-6,
        'gate1_err':1e-5,
        'gate2_err':5e-4,
        'readout_err':1e-3,
        'erasure':0,
    }
)

GoogleIdenticalNoiseParams = copy.deepcopy(GoogleNoiseParams)
GoogleIdenticalNoiseParams.error_stdevs = {
    'T1':0,
    'T2':0,
    'gate1_err':0,
    'gate2_err':0,
    'readout_err':0,
    'erasure':0,
}