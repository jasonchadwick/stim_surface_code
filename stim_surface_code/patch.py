"""A combination of code from Josh and Sophia, with changes and extensions made by Natalia and I
(Jason) to allow for variable qubit error rates. This is a base class that cannot actually
generate a Stim circuit on its own, and is meant to be inherited to create a 

Features:
    T1, T2, measurement error, single qubit error, and two qubit error rates can be set
    on a per-qubit or per-coupler basis.
    Correlation rates between qubits can be specified to model some effects of crosstalk.

"""
from itertools import product, chain, combinations
import numpy as np
import numpy_indexed as npi
from numpy.typing import NDArray
from typing import Any, Callable
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import stim
import sinter
import pymatching
import qc_utils.matplotlib_setup as mpl_setup
import qc_utils.plot as plot_utils
import qc_utils.stats

class Qubit():
    """A single physical qubit on a device.
    """
    def __init__(self, idx: int, coords: tuple[int, int]) -> None:
        """Initialize.
        
        Args:
            idx: Index of the qubit.
            coords: Coordinates of the qubit on the device.
        """
        self.idx: int = idx
        self.coords: tuple[int, int] = coords

    def __repr__(self) -> str:
        return f'{self.idx}, Coords: {self.coords}'

class DataQubit(Qubit):
    """Data qubit used to store logical information.
    """
    pass

class MeasureQubit(Qubit):
    """Ancilla qubit used to perform stabilizer measurements.
    """
    def __init__(self, idx: int, coords: tuple[int, int], data_qubits: list[DataQubit | None], basis: str) -> None:
        """Initialize.
        
        Args:
            idx: Index of the qubit.
            coords: Coordinates of the qubit on the device.
            data_qubits: List of data qubits that this qubit measures.
        """
        super().__init__(idx, coords)
        self.data_qubits = data_qubits
        self.basis = basis

    def __repr__(self):
        return f'{self.idx}, Coords: {self.coords}, Basis: {self.basis}, Data Qubits: {self.data_qubits}'

class SurfaceCodePatch():
    """A Stim code generator for a single rotated planar surface code tile.

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
    def __init__(
            self, 
            dx: int,
            dz: int,
            dm: int,
            gate1_time: float = 25e-9, # params from Google scaling logical qubit paper
            gate2_time: float = 34e-9, 
            meas_time: float = 500e-9, 
            reset_time: float = 250e-9, # Multi-level reset from McEwen et al. (2021)
            apply_idle_during_gates: bool = True,
            save_stim_circuit: bool = False,
            id_offset: int = 0,
        ) -> None:
        """Initializes the instance based on spam preference.

        Args:
            dx: X distance of code (~ number of rows of qubits).
            dz: Z distance of code (~ number of columns of qubits).
            dm: Temporal distance of code (= number of rounds of measurement). 
            gate1_time: duration of single-qubit gates.
            gate2_time: duration of two-qubit gates.
            meas_time: duration of readout.
            reset_time: duration of reset operation.
            apply_idle_during_gates: If True, apply idle errors to qubits while
                performing gate operations and measurement (gate error rates
                should be adjusted accordingly - total gate error will be idle
                error plus gate error rates).
            save_stim_circuit: If True, save each Stim circuit for quick
                querying (if noise models have not changed since last time).
            id_offset: Offset to add to qubit indices.
        """
        self.dx = dx
        self.dz = dz
        self.dm = dm

        self.device: list[list[Qubit | None]] = [
            [None for _ in range(2*self.dz+1)] for _ in range(2*self.dx+1)]
        
        assert len(self.device) == 2*dx+1
        assert len(self.device[0]) == 2*dz+1

        self.data = self.place_data(id_offset)

        (self.logical_x_qubits, 
         self.logical_z_qubits) = self.set_logical_operators()

        # create self.x_ancilla and self.z_ancilla; fill in self.device
        self.place_ancilla(id_offset)

        self.ancilla = self.x_ancilla + self.z_ancilla
        self.all_qubits: list[DataQubit | MeasureQubit] = self.ancilla + self.data
        self.all_qubit_coords = np.array([q.coords for q in self.all_qubits], int)
        self.all_qubit_indices = np.array([q.idx for q in self.all_qubits], int)
        # assert len(self.all_qubits) == 2*(dx*dz)-1
        self.qubit_name_dict = {q.idx:q for q in self.all_qubits}

        self.error_vals: dict[str, dict[int | tuple[int, int], float]] = {
            'T1':{},
            'T2':{},
            'readout_err':{},
            'gate1_err':{},
            'gate2_err':{},
        }
        self.error_vals_initialized = False

        self.gate1_time = gate1_time
        self.gate2_time = gate2_time
        self.meas_time = meas_time
        self.reset_time = reset_time

        # Keeps track of Stim measurement results for each qubit measured in
        # each round. Each list entry is a round, and each key to the inner
        # dictionary is a qubit number. The value is the position of the result
        # in Stim's meas_rec.
        # Right now, this should be reset manually at the start of get_stim().
        # Would be nice to have a more elegant method for doing this.
        self.meas_record: list[dict[int, int]] = []

        self.qubit_pairs: list[tuple[int, int]] = []
        for i in range(4):
            for measure_x in self.x_ancilla:
                dqi = measure_x.data_qubits[i]
                if dqi != None:
                    self.qubit_pairs.append((measure_x.idx, dqi.idx))
            for measure_z in self.z_ancilla:
                dqi = measure_z.data_qubits[i]
                if dqi != None:
                    self.qubit_pairs.append((dqi.idx, measure_z.idx))
        
        self.qubits_to_highlight = []

        # do not consider errors that happen with lower probability than this
        self.probability_cutoff: float = 10**-6 

        self.apply_idle_during_gates = apply_idle_during_gates

        self.correlations = {}
        self.consider_correlations = False

        self.delay_between_rounds = 0

        self.qubit_amplification_repeats: dict[int | tuple[int, int], int] = {}

        # determines which qubits are doing things
        self.qubits_active = {i:True for i in self.all_qubit_indices}

        self.save_stim_circuit = save_stim_circuit
        self.saved_stim_circuit_X: stim.Circuit | None = None
        self.saved_stim_circuit_Z: stim.Circuit | None = None

    def total_gate1_err(self, qubit) -> float:
        """Get the total error rate of single-qubit gates on specified qubit,
        accounting for T1/T2 errors.
        
        Args:
            qubit: Qubit index.
        
        Returns:
            Total single-qubit gate error rate.
        """
        ge = self.error_vals['gate1_err'][qubit] 
        p_x = max(0, 0.25 * (1 - np.exp(-self.gate1_time*1.0 / self.error_vals['T1'][qubit])))
        p_y = p_x
        p_z = max(0, 0.5 * (1 - np.exp(-self.gate1_time*1.0 / self.error_vals['T2'][qubit])) - p_x)
        return p_x+p_y+p_z+ge

    def cycle_time(self) -> float:
        """Get the amount of times it takes to perform a single round of
        stabilizer measurements.
        
        Returns:
            Time for one stabilizer cycle."""
        return self.gate1_time * 2 + self.gate2_time * 4 + self.meas_time + self.reset_time

    def place_data(
            self,
            id_offset: int = 0,
        ) -> list[DataQubit]:
        """Create the device object that will hold all physical qubits, and
        place data qubits within it.

        Args:
            id_offset: Offset to add to qubit indices.
        
        Returns:
            list of DataQubit objects.
        """
        data: list[DataQubit] = [
            DataQubit(id_offset+(self.dz*row + col), (2*row+1, 2*col+1)) 
            for col in range(self.dz) for row in range(self.dx)]
        
        for data_qubit in data:
            self.device[data_qubit.coords[0]][data_qubit.coords[1]] = data_qubit
        return data

    def _get_neighboring_data_qubits(self, coords: tuple[int, int], basis: str) -> list[DataQubit | None]:
        """TODO
        """
        if basis == 'Z':
            offsets = np.array([[-1, -1], [-1, +1], [+1, -1], [+1, +1]])
        else:
            offsets = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])
        qubits = []
        for offset in offsets:
            new_coords = (coords[0] + offset[0], coords[1] + offset[1])
            if (new_coords[0] > 0 and new_coords[0] < len(self.device)
                and new_coords[1] > 0 and new_coords[1] < len(self.device[0])):
                q = self.device[new_coords[0]][new_coords[1]]
                if isinstance(q, DataQubit):
                    qubits.append(q)
                else:
                    qubits.append(None)
            else:
                qubits.append(None)
            
        return qubits

    def place_ancilla(self, id_offset: int = 0) -> None:
        """Place ancilla (non-data) qubits in the patch. Must be run *after*
        place_data.

        Args:
            id_offset: Offset to add to qubit indices.
        """
        # number of qubits already placed (= index of next qubit)
        q_count = len(self.data) + id_offset

        self.x_ancilla: list[MeasureQubit] = []
        self.z_ancilla: list[MeasureQubit] = []
        for row in range(self.dx+1):
            for col in range(self.dz+1):
                if (row + col) % 2 == 1 and col != 0 and col != self.dz: # X basis
                    coords = (2*row, 2*col)
                    data_qubits = self._get_neighboring_data_qubits(coords, 'X')
                    if all(q is None for q in data_qubits):
                        continue
                    measure_q = MeasureQubit(q_count, coords, data_qubits, 'X')
                    self.device[coords[0]][coords[1]] = measure_q
                    self.x_ancilla.append(measure_q)
                    q_count += 1
                elif (row + col) % 2 == 0 and row != 0 and row != self.dx: # Z basis
                    coords = (2*row, 2*col)
                    data_qubits = self._get_neighboring_data_qubits(coords, 'Z')
                    if all(q is None for q in data_qubits):
                        continue
                    measure_q = MeasureQubit(q_count, coords, data_qubits, 'Z')
                    self.device[coords[0]][coords[1]] = measure_q
                    self.z_ancilla.append(measure_q)
                    q_count += 1

    def set_logical_operators(self) -> tuple[set[DataQubit], set[DataQubit]]:
        """Set qubits whose X (Z) operators together correspond to the logical X
        (Z) operators.

        Returns:
            logical_x_qubits: Set of DataQubits whose combined Pauli X product
                yields the X observable.
            logical_z_qubits: Set of DataQubits whose combined Pauli Z product
                yields the Z observable.
        """
        logical_z_qubits: set[DataQubit] = {
            q for q in self.data if q.coords[0] == self.dx}
        logical_x_qubits: set[DataQubit] = {
            q for q in self.data if q.coords[1] == self.dz}
        return logical_x_qubits, logical_z_qubits

    def __repr__(self) -> str:
        output = '-' * 5 * len(self.device[0]) + '-\n'
        for i in range(len(self.device)):
            output += '|'
            for j in range(len(self.device[0])):
                d = self.device[i][j]
                if d is None:
                    output += '    |'
                else:
                    if d.idx in self.qubits_to_highlight:
                        output += color.GRNHB
                        if isinstance(d, MeasureQubit):
                            output += f'{d.basis}{d.idx:3d}'
                        else:
                            if d in self.logical_z_qubits:
                                output += f'L{d.idx:3d}'
                            else:
                                output += f'{d.idx:4d}'
                        output += color.END + '|'
                    else:
                        if isinstance(d, MeasureQubit):
                            if d.basis == 'X':
                                c = color.BLU
                            else:
                                c = color.RED
                            output += c + f'{d.basis}{d.idx:3d}' + color.END + '|'
                        else:
                            output += color.BOLD
                            if d in self.logical_z_qubits:
                                output += f'L{d.idx:3d}'
                            else:
                                output += f'{d.idx:4d}'
                            output += color.END + '|'
            output += '\n--'

            # show qubit connections
            for j in range(len(self.device[0])):
                output += '-' * 2
                topleft = self.device[i][j]
                topright = None if j >= len(self.device[0])-1 else self.device[i][j+1]
                botleft = None if i >= len(self.device)-1 else self.device[i+1][j]
                botright = None if i >= len(self.device)-1 or j >= len(self.device[0])-1 else self.device[i+1][j+1]

                if topleft is not None and botright is not None:
                    q0 = topleft.idx
                    q1 = botright.idx
                    if (q0,q1) in self.qubits_to_highlight or (q1,q0) in self.qubits_to_highlight:
                        output += color.YELHB + '-O-' + color.END
                    else:
                        output += '-+-'
                elif topright is not None and botleft is not None:
                    q0 = topright.idx
                    q1 = botleft.idx
                    if (q0,q1) in self.qubits_to_highlight or (q1,q0) in self.qubits_to_highlight:
                        output += color.YELHB + '-O-' + color.END
                    else:
                        output += '-+-'
                else:
                    output += '---'
                    continue
            output = output[:-1]
            output += '\n'
        return output

    def get_gate1_T1T2_err(self, qubit: int = 0) -> float:
        """Get T1/T2 error rate over the course of a single-qubit gate.
        
        Args:
            qubit: Qubit to get error rate for.
        
        Returns:
            T1/T2 error rate during a single-qubit gate.
        """
        t = self.gate1_time
        p_x = max(0, 0.25 * (1 - np.exp(-t*1.0 / self.error_vals['T1'][qubit])))
        p_y = p_x
        p_z = max(0, 0.5 * (1 - np.exp(-t*1.0 / self.error_vals['T2'][qubit])) - p_x)
        return p_x + p_y + p_z

    def get_gate2_T1T2_err(
            self, 
            qubit_pair: tuple[int, int] | None = None,
        ) -> float:
        """Get T1/T2 error rate over the course of a two-qubit gate.
        
        Args:
            qubit_pair: Qubit pair to get error rate for.
        
        Returns:
            T1/T2 error rate during a two-qubit gate.
        """
        if qubit_pair is None:
            qubit_pair = self.qubit_pairs[0]
        qubit0, qubit1 = qubit_pair
        t = self.gate2_time
        p_x0 = max(0, 0.25 * (1 - np.exp(-t*1.0 / self.error_vals['T1'][qubit0])))
        p_y0 = p_x0
        p_z0 = max(0, 0.5 * (1 - np.exp(-t*1.0 / self.error_vals['T2'][qubit0])) - p_x0)
        p_x1 = max(0, 0.25 * (1 - np.exp(-t*1.0 / self.error_vals['T1'][qubit1])))
        p_y1 = p_x1
        p_z1 = max(0, 0.5 * (1 - np.exp(-t*1.0 / self.error_vals['T2'][qubit1])) - p_x1)
        return p_x0 + p_y0 + p_z0 + p_x1 + p_y1 + p_z1

    def set_error_vals(
            self, 
            error_dict: dict[str, dict[int | tuple[int, int], float]],
        ) -> None:
        """Set physical qubit error rates.

        Args:
            dict: dictionary mapping error term names to dictionaries of values
                for each qubit or qubit pair. Must contain terms for all error
                types and all qubits.
        """
        assert all([idx in error_dict['T1'] for idx in self.all_qubit_indices])
        assert all([idx in error_dict['T2'] for idx in self.all_qubit_indices])
        assert all([idx in error_dict['readout_err'] 
                    for idx in self.all_qubit_indices])
        assert all([idx in error_dict['gate1_err'] 
                    for idx in self.all_qubit_indices])
        assert all([qubit_pair in error_dict['gate2_err'] 
                    for qubit_pair in self.qubit_pairs])
        assert all([idx in error_dict['erasure']
                    for idx in self.all_qubit_indices])
        
        self.error_vals = error_dict
        self._error_vals_numpy = {}
        for k,v in self.error_vals.items():
            # sort by index (or, for qubit pairs, by index in self.qubit_pairs)
            if k == 'gate2_err':
                self._error_vals_numpy[k] = np.array([v[pair] for pair in self.qubit_pairs])
            else:
                self._error_vals_numpy[k] = np.array([v[idx] for idx in self.all_qubit_indices])

        self.error_vals_initialized = True
        self.saved_stim_circuit_X = None
        self.saved_stim_circuit_Z = None
    
    def set_error_vals_normal(
            self,
            mean_dict: dict[str, float],
            stdev_dict: dict[str, float] = {
                'T1': 0,
                'T2': 0,
                'readout_err': 0,
                'gate1_err': 0,
                'gate2_err': 0,
                'erasure': 0,
            },
            distributions_log: dict[str, bool] = {
                'T1': False,
                'T2': False,
                'readout_err': True,
                'gate1_err': True,
                'gate2_err': True,
                'erasure': True,
            },
        ) -> None:
        """Set qubit error parameters by drawing from normal distributions. T1
        and T2 are given in units of seconds. Gate error rates are given as
        probability of additional depolarizing error per operation (in addition
        to T1/T2 errors).

        By default, T1 and T2 are sampled from normal distributions. Error rates
        are sampled from lognormal distributions.

        TODO: remove; redundant now that NoiseParams object can set error values
        in the same way. Maybe replace with set_error_vals_from_params?

        Args:
            mean_dict: Dictionary of mean values for each parameter. Must have
                entries for T1, T2, readout_err, gate1_err, and gate2_err.
            stdev_dict: Dictionary of standard deviation values for each
                parameter.
            distributions_log: Whether each value's normal distribution is on a
                log scale. Defaults to True for readout_err, gate1_err, and
                gate2_err.
        """
        assert all([k in mean_dict for k in ['T1', 'T2', 'readout_err', 'gate1_err', 'gate2_err']])
        assert all([k in stdev_dict for k in ['T1', 'T2', 'readout_err', 'gate1_err', 'gate2_err']])
        assert all([k in distributions_log for k in ['T1', 'T2', 'readout_err', 'gate1_err', 'gate2_err']])

        maxvals = {
            'T1': np.inf,
            'T2': np.inf,
            'readout_err': 1.0,
            'gate1_err': 3/4,
            'gate2_err': 15/16,
            'erasure': 1.0
        }
        minvals = {
            'T1': 0.0,
            'T2': 0.0,
            'readout_err': 0.0,
            'gate1_err': 0.0,
            'gate2_err': 0.0,
            'erasure': 0.0
        }

        assert all([v >= minvals[k] for k,v in mean_dict.items()]), 'Mean values below minimum'
        assert all([v <= maxvals[k] for k,v in mean_dict.items()]), 'Mean values above maximum'
        assert all([v >= 0 for v in stdev_dict.values()]), 'Standard deviations must be nonnegative'

        error_val_dict_keys = {
            'T1': self.all_qubit_indices.tolist(),
            'T2': self.all_qubit_indices.tolist(),
            'readout_err': self.all_qubit_indices.tolist(),
            'gate1_err': self.all_qubit_indices.tolist(),
            'gate2_err': self.qubit_pairs,
            'erasure': self.all_qubit_indices.tolist(),
        }
        
        error_vals = {}
        for k,mean in mean_dict.items():
            if distributions_log[k]:
                vals = np.clip(qc_utils.stats.lognormal(mean, stdev_dict[k], size=len(error_val_dict_keys[k])), minvals[k], maxvals[k])
            else:
                vals = np.clip(np.random.normal(mean, stdev_dict[k], size=len(error_val_dict_keys[k])), minvals[k], maxvals[k])
            error_vals[k] = {k:vals[i] for i,k in enumerate(error_val_dict_keys[k])}

        self.set_error_vals(error_vals)

    def apply_operations(self, circ, operation, targets, params):
        """Apply a list of stim circuit operations, aiming to minimize the
        number of (relatively expensive) circuit.append instructions by
        combining operations with the same parameters.
        
        Args:
            circ: Circuit to append operations to.
            operation: Operation to apply.
            targets: List of targets for the operation.
            params: List of parameters for the operation.
        """
        if len(targets) == 0:
            return
        if len(params.shape) == 1:
            params = params[:, None]
        assert len(params.shape) == 2
        assert params.shape[0] == len(targets)
        unique_params = np.unique(params, axis=0)
        for p in unique_params:
            ts = targets[np.all(params == p, axis=1)]
            circ.append(operation, ts.flatten(), p)
        
    def apply_exclusive_errors(
            self, 
            circ: stim.Circuit, 
            qubits: list[int], 
            paulis_list: list[str], 
            ps: list[float], 
        ) -> None:
        """Apply probabilistic correlated errors to a set of qubits. Uses `ELSE_CORRELATED_ERROR`, so 
        only one of the given Pauli errors will happen on any particular sample. Each probability 
        is associated with one list of Pauli operators. All probabilities in `ps` must sum to a value
        less than 1. 

        Example (apply the XXX operator with probability 0.1 or the XYY operator with probability 0.05):
        ```
        apply_errors(circ, [0, 1, 2], ['XXX', 'XYY'], [0.1, 0.05])
        ```

        Args:
            circ: Circuit to append errors to.
            qubits: List of qubits to calculate errors on.
            paulis_list: List of potential Pauli errors to apply. Each entry is
                a string of length equal to len(qubits).
            ps: Probabilities for each Pauli error. Must have same length as
                paulis_list. 
        """
        assert(len(paulis_list) == len(ps))
        remaining_prob = 1
        did_first = False
        for i,paulis in enumerate(paulis_list):
            assert(len(paulis) == len(qubits))
            p = ps[i]
            if not all(p == 'I' for p in paulis):
                targets = []
                for j,q in enumerate(qubits):
                    pauli = paulis[j]
                    if pauli != 'I':
                        if pauli == 'X':
                            target = stim.target_x(q)
                        elif pauli == 'Y':
                            target = stim.target_y(q)
                        else:
                            assert pauli == 'Z'
                            target = stim.target_z(q)
                    
                        if len(targets) == 0:
                            targets.append(target)
                        else:
                            targets.append(stim.target_combiner())
                            targets.append(target)
                circ.append('ELSE_CORRELATED_ERROR' if did_first else 'CORRELATED_ERROR', targets, p / remaining_prob)
                remaining_prob -= p
                did_first = True

    def apply_exclusive_errors_with_correlations(
            self, 
            circ: stim.Circuit, 
            qubits: list[int], 
            paulis_list: list[str], 
            ps: list[float], 
            cutoff: float = 0,
        ) -> None:
        """If an error of type `paulis` occurs on qubit `qubits`, we want to
        probabilistically apply the same error on all qubits that experience
        correlated errors with any of `qubits` (based on
        `self.error_correlations`). Each correlation is treated independently. 

        Args:
            circ: Stim circuit to append errors to.
            qubits: List of qubits to calculate errors on.
            paulis_list: List of potential Pauli errors to apply. Each entry is
                a string of length equal to len(qubits).
            ps: Probabilities for each Pauli error. Must have same length as
                paulis_list. 
            cutoff: Do not apply an error if the probability is below this
                value.
        """

        if not self.consider_correlations:
            self.apply_exclusive_errors(circ, qubits, paulis_list, ps)
            return

        # enumerate all possible pairs that can experience correlated errors
        new_qubits = [q for q in qubits]
        possible_individual_correlations = []
        individual_correlation_probs = []
        for q in qubits:
            corr_qubits = self.correlations[q] if q in self.correlations else []
            for corr_qubit, corr_prob in corr_qubits:
                if corr_qubit not in new_qubits:
                    new_qubits.append(corr_qubit)
                pair = (min(q, corr_qubit), max(q, corr_qubit))
                if pair in possible_individual_correlations:
                    idx = possible_individual_correlations.index(pair)
                    #TODO: is this right?
                    individual_correlation_probs[idx] = 1 - (1-individual_correlation_probs[idx])*(1-corr_prob) 
                else:
                    possible_individual_correlations.append(pair)
                    individual_correlation_probs.append(corr_prob)
        def powerset(iterable):
            'From https://stackoverflow.com/a/1482316/14797949'
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        # calculate relative probabilities of each combination of correlated errors
        possible_combined_correlations = []
        combined_correlation_probs = []
        for corrs in powerset(possible_individual_correlations):
            prob = 1
            for j,corr in enumerate(possible_individual_correlations):
                p = individual_correlation_probs[j]
                if corr in corrs:
                    prob *= p
                else:
                    prob *= (1-p)
            possible_combined_correlations.append(corrs)
            combined_correlation_probs.append(prob)
        assert(np.isclose(1, sum(combined_correlation_probs)))

        def pauli_mul(paulis_1, paulis_2):
            res = ''
            ref_paulis = ['X','Y','Z']
            for p1,p2 in zip(paulis_1, paulis_2):
                if p1 == 'I':
                    res += p2
                elif p2 == 'I':
                    res += p1
                elif p1 == p2:
                    res += 'I'
                else:
                    res += list(set(ref_paulis).difference({p1, p2}))[0]
            return res

        new_pauli_list = []
        new_ps = []
        for i,paulis in enumerate(paulis_list):
            base_paulis = paulis + 'I'*(len(new_qubits) - len(qubits))
            p = ps[i]
            for j,correlation_combination in enumerate(possible_combined_correlations):
                prob = p*combined_correlation_probs[j]
                if prob > cutoff:
                    resulting_paulis = base_paulis
                    for (q0,q1) in correlation_combination:
                        corr_paulis = 'I'*len(new_qubits)
                        i0 = new_qubits.index(q0)
                        i1 = new_qubits.index(q1)
                        corr_paulis = corr_paulis[:i1] + base_paulis[i0] + corr_paulis[i1+1:] # q1 gets the error of q0
                        corr_paulis = corr_paulis[:i0] + base_paulis[i1] + corr_paulis[i0+1:] # q0 gets the error of q1
                        resulting_paulis = pauli_mul(resulting_paulis, corr_paulis)
                    new_pauli_list.append(resulting_paulis)
                    new_ps.append(prob)
        self.apply_exclusive_errors(circ, new_qubits, new_pauli_list, new_ps)

    def apply_idle(
            self, 
            circ: stim.Circuit, 
            t: float, 
            non_idle_qubits: list[int] = [], 
            idle_qubits: list[int] = [],
        ) -> None:
        """Applies idle errors based on duration t and preconfigured qubit T1
        and T2 times.

        TODO: maybe do not apply idle errors on qubits that have just been reset?

        Args:
            circ: Stim circuit to append to.
            t: Amount of time to idle.
            non_idle_qubits: List of qubit indices which do NOT experience the
                idle errors. Only used if idle_qubits = [].
            idle_qubits: List of qubit indices which DO experience the idle
                errors. If empty, set to list of ALL qubits (except those in
                non_idle_qubits). 
        """
        if idle_qubits==[]:
            idle_qubits = [idx for idx in self.all_qubit_indices if idx not in non_idle_qubits]
        idle_qubits = [q for q in idle_qubits]

        # todo: vectorize with numpy
        targets = np.array(idle_qubits)[:,None]
        params = np.zeros((len(idle_qubits), 3))
        for i,q in enumerate(idle_qubits):
            p_x = max(0, 0.25 * (1 - np.exp(-t*1.0 / self.error_vals['T1'][q])))
            p_y = p_x
            p_z = max(0, 0.5 * (1 - np.exp(-t*1.0 / self.error_vals['T2'][q])) - p_x)
            params[i] = [p_x, p_y, p_z]
        self.apply_operations(circ, 'PAULI_CHANNEL_1', targets, params)
        
    def apply_1gate(
            self, 
            circ: stim.Circuit, 
            gate: str, 
            qubits: list[int],
            unused_qubits_idle: bool = True,
        ) -> None:
        """Applies a single-qubit gate to qubits. Applies idle errors to unused
        qubits. Adds a TICK instruction after.

        Args:
            circ: Stim circuit to append to.
            gate: Single-qubit gate to perform on each qubit, e.g. 'X' or 'H'.
            qubits: List of qubits to apply the gate to.
            unused_qubits_idle: If True, apply idle errors to unused qubits.
                Appends a TICK instruction after.
        """
        qubits = [q for q in qubits if self.qubits_active[q]]
        qubit_repeat_nums = dict()
        for qubit in qubits:
            if qubit in self.qubit_amplification_repeats:
                qubit_repeat_nums[qubit] = self.qubit_amplification_repeats[qubit]
            else:
                qubit_repeat_nums[qubit] = 1
        while len(qubit_repeat_nums) > 0:
            qubits_to_use = list(qubit_repeat_nums.keys())
            circ.append(gate, qubits_to_use)
            if self.consider_correlations:
                for qubit in qubits_to_use:
                    p = self.error_vals['gate1_err'][qubit]
                    # Apply depolarizing (equal probability of any Pauli error)
                    self.apply_exclusive_errors_with_correlations(circ, [qubit], ['X','Y','Z'], [p/3, p/3, p/3], cutoff=self.probability_cutoff)
            else:
                targets = np.array(qubits_to_use)
                target_indices = npi.indices(self.all_qubit_indices, targets, axis=0)
                ps = self._error_vals_numpy['gate1_err'][target_indices]
                self.apply_operations(circ, 'DEPOLARIZE1', targets, ps)

            if self.apply_idle_during_gates and unused_qubits_idle:
                self.apply_idle(
                    circ, 
                    self.gate1_time,
                )
                circ.append('TICK')
            elif self.apply_idle_during_gates:
                self.apply_idle(
                    circ, 
                    self.gate1_time, 
                    idle_qubits=qubits_to_use,
                )
            elif unused_qubits_idle:
                self.apply_idle(
                    circ, 
                    self.gate1_time, 
                    non_idle_qubits=qubits_to_use,
                )
                circ.append('TICK')

            # subtract 1 from all repeat nums and remove elems with 0 more repeats
            qubit_repeat_nums = {q:r-1 for q,r in qubit_repeat_nums.items() if r > 1}

    def apply_2gate(
            self, 
            circ: stim.Circuit, 
            gate: str, 
            qubit_pairs: list[tuple[int, int]],
            unused_qubits_idle: bool = True,
        ) -> None:
        """Applies a two-qubit gate to qubit_pairs. Applies idle errors to
        unused qubits. Adds a TICK instruction after.

        Args:
            circ: Stim circuit to append to.
            gate: Two-qubit gate to perform on each qubit, e.g. 'CX'.
            qubit_pairs: List of qubit pairs to apply the gate to.
        """
        qubit_pairs = [(q0,q1) for (q0,q1) in qubit_pairs if self.qubits_active[q0] and self.qubits_active[q1]]
        pair_repeat_nums = dict()
        for pair in qubit_pairs:
            if pair in self.qubit_amplification_repeats:
                pair_repeat_nums[pair] = self.qubit_amplification_repeats[pair]
            else:
                pair_repeat_nums[pair] = 1
        while len(pair_repeat_nums) > 0:
            pairs_to_use = list(pair_repeat_nums.keys())
            qubits = []
            for q1,q2 in pairs_to_use:
                qubits += [q1, q2]
            circ.append(gate, qubits)
            if self.consider_correlations:
                for q1,q2 in pairs_to_use:
                    p = self.error_vals['gate2_err'][q1,q2]
                    # Apply depolarizing (equal probability of any two-qubit Pauli error)
                    pauli_strings = [ps[0]+ps[1] for ps in product(['I','X','Y','Z'], repeat=2) if ps != ('I','I')]
                    self.apply_exclusive_errors_with_correlations(circ, [q1,q2], pauli_strings, [p/15]*15, cutoff=self.probability_cutoff)
            else:
                targets = np.array(qubit_pairs)
                target_indices = npi.indices(np.array(self.qubit_pairs), targets, axis=0)
                ps = self._error_vals_numpy['gate2_err'][target_indices]
                self.apply_operations(circ, 'DEPOLARIZE2', targets, ps)
                                                            
            if self.apply_idle_during_gates and unused_qubits_idle:
                self.apply_idle(
                    circ, 
                    self.gate2_time,
                )
                circ.append('TICK')
            elif self.apply_idle_during_gates:
                self.apply_idle(
                    circ, 
                    self.gate2_time, 
                    idle_qubits=qubits,
                )
            elif unused_qubits_idle:
                self.apply_idle(
                    circ, 
                    self.gate2_time, 
                    non_idle_qubits=qubits,
                )
                circ.append('TICK')

            # subtract 1 from all repeat nums and remove elems with 0 more repeats
            pair_repeat_nums = {p:r-1 for p,r in pair_repeat_nums.items() if r > 1}

    def apply_erasures(
            self,
            circ: stim.Circuit,
            qubits: list[int],
        ):
        """TODO
        """
        if self._error_vals_numpy['erasure'].sum() == 0:
            return
        for q in qubits:
            circ.append('HERALDED_ERASE', q, self.error_vals['erasure'][q])
            circ.append('DETECTOR', stim.target_rec(-1), self.qubit_name_dict[q].coords + (0,1))
        # for q in qubits:
        #     erasure_ancilla_idx = len(self.all_qubits) + q
        #     circ.append('R', erasure_ancilla_idx)
        #     if np.random.rand() < self.error_vals['erasure'][q]:
        #         circ.append('DEPOLARIZE1', q, 3/4)
        #         circ.append('X_ERROR', erasure_ancilla_idx, 0.99)
        #     circ.append('M', erasure_ancilla_idx)
        #     circ.append('DETECTOR', stim.target_rec(-1), self.qubit_name_dict[q].coords + (0,1))

        # Update measurement record indices
        for round in self.meas_record:
            for q, idx in round.items():
                round[q] = idx - len(qubits)

    def apply_reset(
            self,
            circ: stim.Circuit,
            qubits: list[int],
            unused_qubits_idle: bool = True,
        ):
        """Applies a Z reset operator to specified qubits.
        
        Args:
            circ: Stim circuit to append to.
            qubits: List of qubits to apply the operation to.
            unused_qubits_idle: If True, apply idle errors to unused qubits.
                Appends a TICK instruction after. 
        """

        circ.append('R', qubits)

        if unused_qubits_idle:
            self.apply_idle(
                circ, 
                self.reset_time, 
                non_idle_qubits=qubits
            )
            circ.append('TICK')

    def apply_meas(
            self,
            circ: stim.Circuit,
            qubits: list[int],
            unused_qubits_idle: bool = True,
        ):
        """Applies a Z measurement operator to specified qubits. Updates
        self.meas_record.
        
        Args:
            circ: Stim circuit to append to.
            qubits: List of qubits to apply the operation to.
            unused_qubits_idle: If True, apply idle errors to unused qubits.
                Appends a TICK instruction after. 
        """

        if self.apply_idle_during_gates:
            self.apply_idle(
                circ, 
                self.meas_time, 
                non_idle_qubits=qubits
            )
        if unused_qubits_idle:
            self.apply_idle(
                circ, 
                self.meas_time, 
                non_idle_qubits=qubits
            )
            circ.append('TICK')
        
        for qubit in qubits:
            circ.append('M', qubit, self.error_vals['readout_err'][qubit])

        # Update measurement record indices
        meas_round = {}
        for i in range(len(qubits)):
            q = qubits[-(i + 1)]
            meas_round[q] = -(i + 1)
        for round in self.meas_record:
            for q, idx in round.items():
                round[q] = idx - len(qubits)
        self.meas_record.append(meas_round)

    def meas_ideal(
            self,
            circ: stim.Circuit,
            op: str,
            qubits: list[int],
        ) -> None:
        """Applies ideal M or MX operation to qubits.
        
        Args:
            circ: Stim circuit to append to.
            op: Measure/reset op to apply, such as 'M' or 'MX'.
            qubits: List of qubits to apply the operation to.    
        """
        for qubit in qubits:
            circ.append(op, qubit)

        # Update measurement record indices
        meas_round = {}
        for i in range(len(qubits)):
            q = qubits[-(i + 1)]
            meas_round[q] = -(i + 1)
        for round in self.meas_record:
            for q, idx in round.items():
                round[q] = idx - len(qubits)
        self.meas_record.append(meas_round)

    def get_meas_rec(
            self, 
            round_idx: int, 
            qubit: int
        ) -> stim.GateTarget:
        """Return the Stim measurement target associated with a particular qubit
        in a particular round. Used when setting detector targets in
        self.syndrome_round.
        
        Args:
            round_idx: round index of desired measurement.
            qubit: qubit of interest.
        
        Returns:
            Stim measurement record target (output of stim.target_rec)
        """
        return stim.target_rec(self.meas_record[round_idx][qubit])

    def initialize_ideal(self, circ: stim.Circuit, state: str) -> None:
        """Initialize patch noiselessly in some target state.
        
        Args:
            circ: Stim circuit to append operations to.
            state: One of '0', '1', '+', '-', 'i', '-i'.
        """
        all_qubits = self.all_qubit_indices.tolist()
        data_qubits = [q.idx for q in self.data]

        circ.append('R', all_qubits)

        if state == '0':
            return
        elif state == '1':
            circ.append('X', data_qubits)
        elif state == '+':
            circ.append('H', data_qubits)
        elif state == '-':
            circ.append('X', data_qubits)
            circ.append('H', data_qubits)
        elif state == 'i':
            circ.append('H', data_qubits)
            circ.append('S', data_qubits)
        elif state == '-i':
            circ.append('X', data_qubits)
            circ.append('H', data_qubits)
            circ.append('S', data_qubits)
        else:
            raise Exception(f'Initialization state "{state}" not supported')
        circ.append('TICK')

    def measure_logical_operator_ideal(
            self, 
            circ: stim.Circuit, 
            basis: str,
            add_observable: bool = False
        ) -> None:
        """Perform an ideal measurement of the logical X, Y, or Z operator.

        Requires that self.logical_x_qubits and self.logical_z_qubits are
        defined and correct. Current implementation adds a DETECTOR to track the
        measurement result, not an OBSERVABLE_INCLUDE (because stim only allows
        deterministic observables, and this is not necessarily the case here.)

        Args:
            circ: circuit on which to append measurement operations.
            basis: 'X', 'Y', or 'Z'.
            add_observable: if True, add an OBSERVABLE to Stim. Otherwise, use a
                detector instead.
        """
        raise NotImplementedError
    
        assert len(self.logical_z_qubits) != 0
        assert len(self.logical_x_qubits) != 0

        measured_qubits: list[tuple[int, str]] = []

        if basis == 'X':
            measured_qubits = [(q.idx,'X') for q in self.logical_x_qubits]
        elif basis == 'Z':
            measured_qubits = [(q.idx,'Z') for q in self.logical_z_qubits]
        else:
            assert basis == 'Y'
            for xq in self.logical_x_qubits:
                if xq in self.logical_z_qubits:
                    measured_qubits.append((xq.idx, 'Y'))
                else:
                    measured_qubits.append((xq.idx, 'X'))
            
            for zq in self.logical_z_qubits:
                if zq not in self.logical_x_qubits:
                    measured_qubits.append((zq.idx, 'Z'))

        stim_targets: list[stim.GateTarget] = []
        for qubit, basis in measured_qubits:
            if basis == 'X':
                stim_targets.append(stim.target_x(qubit))
                stim_targets.append(stim.target_combiner())
            elif basis == 'Y':
                stim_targets.append(stim.target_y(qubit))
                stim_targets.append(stim.target_combiner())
            else:
                assert basis == 'Z'
                stim_targets.append(stim.target_z(qubit))
                stim_targets.append(stim.target_combiner())
        # remove last target_combiner
        stim_targets = stim_targets[:-1]
        circ.append('MPP', stim_targets)

        if add_observable:
            circ.append('OBSERVABLE_INCLUDE', stim.target_rec(-1), 0)
        else:
            circ.append('DETECTOR', stim.target_rec(-1))

    def syndrome_round(
            self, 
            circ: stim.Circuit, 
            deterministic_detectors: list[int] = [],
            inactive_detectors: list[int] = [],
        ) -> stim.Circuit:
        """Add stim code for one syndrome round to input circuit. Uses
        self.active_qubits to decide which qubits to apply operations to.

        Args:
            circ: Stim circuit to add to.
            deterministic_detectors: list of ancilla whose measurements are
                (ideally, with no errors) deterministic.
            inactive_detectors: list of detectors to NOT enforce this round (but
                stabilizer is still measured)
        
        Returns:
            Modified Stim circuit with a syndrome round appended.
        """
        self.apply_reset(circ, [measure.idx for measure in self.ancilla
                                if self.qubits_active[measure.idx]])

        # CNOTs
        self.apply_1gate(circ, 'H', [measure.idx 
                                     for measure in self.x_ancilla
                                     if self.qubits_active[measure.idx]])
        for i in range(4):
            err_qubits = []
            for measure in self.x_ancilla:
                if self.qubits_active[measure.idx]:
                    dqi = measure.data_qubits[i]
                    if dqi != None:
                        err_qubits += [(measure.idx, dqi.idx)]
            for measure in self.z_ancilla:
                if self.qubits_active[measure.idx]:
                    dqi = measure.data_qubits[i]
                    if dqi != None:
                        err_qubits += [(dqi.idx, measure.idx)]
            self.apply_2gate(circ,'CX',err_qubits)
        self.apply_1gate(circ, 'H', [measure.idx 
                                     for measure in self.x_ancilla
                                     if self.qubits_active[measure.idx]])

        self.apply_erasures(circ, [q.idx for q in self.all_qubits])

        # Measure
        self.apply_meas(circ, [measure.idx for measure in self.ancilla
                               if self.qubits_active[measure.idx]])

        for ancilla in self.ancilla:
            if (self.qubits_active[ancilla.idx] 
                and ancilla.idx not in inactive_detectors):
                if ancilla.idx in deterministic_detectors:
                    # no detector history to compare
                    circ.append(
                        'DETECTOR', 
                        self.get_meas_rec(-1, ancilla.idx), 
                        ancilla.coords + (0,)
                    )
                else:
                    # compare detector to a previous round
                    circ.append(
                        'DETECTOR', 
                        [self.get_meas_rec(-1, ancilla.idx),
                        self.get_meas_rec(-2, ancilla.idx)],
                        ancilla.coords + (0,)
                    )

        circ.append('TICK')
        circ.append('SHIFT_COORDS', [], [0.0, 0.0, 1.0])

        return circ
    
    def get_stim(self) -> stim.Circuit:
        """To be implemented by child class.
        """
        self.meas_record: list[dict[int, int]] = []
        raise NotImplementedError
    
    def count_logical_errors(
            self, 
            shots: int = 10**7, 
            fractional_stdev: float = 0.01,
            batch_size: int = 10**5, 
            max_errors: int | None = None,
            use_sinter: bool = True,
            task_kwargs: dict[str, Any] = {},
            num_workers: int = 6,
            **stim_kwargs,
        ) -> tuple[float, int]:
        """Simulate the Stim circuit and calculate the empirical logical error
        rate.
        
        Args:
            shots: Number of shots to take. Should be larger than
                1/(expected_err_rate).
            fractional_stdev: If not using Sinter, terminate early if standard
                deviation of observed error rate is less than this fraction of
                the error rate.
            batch_size: If not using Sinter, number of shots to batch together.
            max_errors: If using Sinter, terminate early if we see this many
                errors.
            use_sinter: If True, simulate in Sinter.
            task_kwargs: If using Sinter, additional arguments to pass to the
                sinter.Task constructor.
            num_workers: If using Sinter, number of parallel workers to use.
        
        Returns:
            err_rate: Observed logical error rate.
            stdev: Standard deviation of observed rate.
            completed_shots: Number of simulation shots completed.
        """
        circuit = self.get_stim(**stim_kwargs)

        err_rate: float = 0
        if use_sinter:
            task = sinter.Task(
                circuit=circuit, 
                **task_kwargs,
            )
            stats = sinter.collect(
                num_workers=num_workers,
                tasks=[task],
                decoders=(['pymatching'] if 'decoders' not in task_kwargs else None),
                max_shots=(shots if 'max_shots' not in task_kwargs else None),
                max_errors=(max_errors if 'max_errors' not in task_kwargs else None),
            )[0]
            num_errors = stats.errors
            completed_shots = stats.shots
            err_rate = num_errors / completed_shots
            completed_shots = completed_shots
        else:
            # From PyMatching docs: "Error instruction in the DEM that cause more than two detection events and do not have a suggested decomposition into edges are ignored."
            matching = pymatching.Matching.from_detector_error_model(detector_error_model)
            sampler = circuit.compile_detector_sampler()

            total_errors = 0
            completed_shots = 0
            while shots > 0:
                shots_to_take = min(batch_size, shots)
                syndrome, actual_observables = sampler.sample(shots=shots_to_take, separate_observables=True)

                predicted_observables = matching.decode_batch(syndrome)
                num_errors = np.sum(np.any(predicted_observables != actual_observables, axis=1))

                total_errors += num_errors
                completed_shots += shots_to_take
                shots -= shots_to_take

                err_rate = total_errors / completed_shots
                stdev = np.sqrt(err_rate * (1 - err_rate) / completed_shots)
                if np.all(stdev < err_rate * fractional_stdev):
                    break
            if shots == 0:
                print('Reached max shot limit.')

        return err_rate, completed_shots

    def count_detection_events(
            self,
            shots: int, 
            only_intermediate_detectors: bool = True,
            return_full_data: bool = False,
            **stim_kwargs,
        ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Count detection rate for each detector in circuit. Shots are taken until all standard
        deviations are below `fractional_stdev` or until reaching the maximum number specified by
        `shots`.

        Args:
            shots: Number of shots to take. Should be larger than
                1/(expected_err_rate).
            only_intermediate_detectors: If True, ignore the initial and final
                detectors and only return counts for mid-circuit detectors. This
                option also reshapes the returned arrays into shape
                (self.num_rounds-1, detectors_per_round).
            return_full_data: If True, return individual shot data.
            stim_kwargs: Arguments to pass on to self.get_stim().

        Returns:
            fractions: Rate that each detector signaled.
            detector_samples: If return_full_data, a list of observed detector
                signals for each shot. Shape is (completed_shots, num_detectors)
                if only_intermediate_detectors is False, and (completed_shots,
                self.dm-2, detectors_per_round) otherwise.
            observable_samples: If return_full_data, a list of observed
                observables for each shot. Shape is (completed_shots,
                num_observables).
        """
        circuit = self.get_stim(**stim_kwargs)

        sampler = circuit.compile_detector_sampler()

        detector_samples, observable_samples = sampler.sample(shots=shots, separate_observables=True)

        totals = np.sum(detector_samples, axis=0, dtype=np.uint64)
        fractions = totals / shots
        
        if only_intermediate_detectors:
            num_detectors = circuit.num_detectors

            detector_round_indices = np.array(list(circuit.get_detector_coordinates().values()))[:,2]
            init_detector_count = np.sum(detector_round_indices == 0)
            final_detector_count = np.sum(detector_round_indices == self.dm)
            intermediate_detector_count = len(detector_round_indices) - init_detector_count - final_detector_count
            detectors_per_round = intermediate_detector_count // (self.dm-1)
            assert(intermediate_detector_count % (self.dm-1) == 0)

            fractions = np.reshape(fractions[init_detector_count : num_detectors-final_detector_count], (self.dm-1, detectors_per_round))
            detector_samples = np.reshape(detector_samples[:, init_detector_count : num_detectors-final_detector_count], (shots, self.dm-1, detectors_per_round))

        if return_full_data:
            return fractions, detector_samples, observable_samples
        return fractions, None, None

    def get_sinter_task(
            self, 
            task_kwargs: dict = {},
            **stim_kwargs
        ) -> sinter.Task:
        """Return a Sinter task for the current circuit.

        Args:
            task_kwargs: Dictionary of additional arguments to pass to the
                sinter.Task constructor, such as json_metadata or decoder.
            stim_kwargs: Arguments to pass on to self.get_stim().
        
        Returns:
            Sinter task for the current circuit.
        """
        circuit = self.get_stim(**stim_kwargs)

        return sinter.Task(
            circuit=circuit, 
            **task_kwargs,
        )

    def get_syndrome_qubits(
            self, 
            only_intermediate_detectors: bool = True, 
            **stim_kwargs,
        ) -> list[Qubit]:
        """Get qubits corresponding to detectors in each syndrome round.
        
        Args:
            only_intermediate_detectors: If True, ignore the initial and final
                detectors and only return counts for mid-circuit detectors. This
                option also reshapes the returned arrays into shape
                (self.num_rounds-1, detectors_per_round).
            stim_kwargs: Arguments to pass on to self.get_stim().
        """
        detector_coords = self.get_stim(**stim_kwargs).get_detector_coordinates()
        round_1_coords = {det: coords[:2] for det, coords in detector_coords.items() if coords[2] == 1}
        syndrome_qubits: list[Qubit | None] = [None] * len(round_1_coords)
        min_idx = min(round_1_coords.keys())
        for det, det_coords in round_1_coords.items():
            qubit = self.device[int(det_coords[0])][int(det_coords[1])]
            syndrome_qubits[det - min_idx] = qubit
        assert None not in syndrome_qubits
        return syndrome_qubits

    def plot_qubit_vals(
            self,
            qubit_vals: list[float] | NDArray[np.float_] | None = None,
            qubit_colors: list[tuple[float | int, ...]] | None = None,
            ax: plt.Axes | None = None,
            plot_text: str = 'idx',
            val_fmt_fn: Callable[[float], str] | None = None,
            cmap_name: str = 'viridis',
            font_size: int = 12,
            vmin: float | None = None,
            vmax: float | None = None,
            norm: mpl.colors.Normalize = mpl.colors.Normalize,
            cbar: mpl.colorbar.Colorbar | None = None,
            cbar_kwargs: dict[str, Any] = {},
        ) -> tuple[plt.Axes, mpl.colorbar.Colorbar | None]:
        """Plot qubit values as a heatmap.

        If neither qubit_vals nor qubit_colors are given, plot qubit colors
        based on data / X ancilla / Z ancilla.

        Args:
            qubit_vals: Array of qubit values to plot.
            qubit_colors: Array of colors to plot qubits with. Overrides
                qubit_vals.
            ax: Axes to plot on. If None, create new figure.
            plot_text: If 'idx', plot qubit indices. If 'val', plot qubit vals.
                If 'none', do not plot text.
            val_fmt_fn: Function to format qubit values as strings for plotting.
                If None, format in scientific notation.
            cmap_name: Name of matplotlib colormap to use.
            font_size: Font size for text.
            vmin: Minimum value for colormap. If None, use min(qubit_vals).
            vmax: Maximum value for colormap. If None, use max(qubit_vals).
            norm: Normalization function for colormap.
            cbar: If given, use this colorbar instead of creating a new one.
            cbar_kwargs: Additional arguments to pass to qc_utils.plot.add_cbar.

        Returns:
            ax: Axes containing plot.
            cbar: Colorbar associated with plot, or None if qubit_vals is None.
        """
        xlims = (-1, 2*self.dz+2)
        ylims = (-1, 2*self.dx+2)

        if ax is None:
            fig,ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()
        assert ax is not None
        
        ax.invert_yaxis()
        ax.set_aspect('equal')

        if qubit_colors is None:
            if qubit_vals is None:
                qubit_colors = [(0.0, 0.0, 0.0, 0.0)] * len(self.all_qubits)
                for i,qubit in enumerate(self.all_qubits):
                    if isinstance(qubit, DataQubit):
                        qubit_colors[qubit.idx] = (1.0,1.0,1.0,1.0)
                    else:
                        assert isinstance(qubit, MeasureQubit)
                        if qubit.basis == 'X':
                            qubit_colors[qubit.idx] = mpl_setup.hex_to_rgb(mpl_setup.colors[0], True)
                        else:
                            qubit_colors[qubit.idx] = mpl_setup.hex_to_rgb(mpl_setup.colors[1], True)
            else:
                vmin = min(qubit_vals) if vmin is None else vmin
                vmax = max(qubit_vals) if vmax is None else vmax
                if cbar is None:
                    cbar = plot_utils.add_cbar(ax, norm(vmin=vmin, vmax=vmax), cmap_name, **cbar_kwargs)
                cmap = mpl.colormaps[cmap_name]
                qubit_colors = []
                for i,val in enumerate(qubit_vals):
                    qubit_colors.append(cmap((val-vmin)/(vmax-vmin)))

        for i,color in enumerate(qubit_colors):
            q = self.qubit_name_dict[i]
            coords = q.coords
            xvals = [coords[1], coords[1]+1, coords[1], coords[1]-1]
            yvals = [coords[0]+1, coords[0], coords[0]-1, coords[0]]

            ax.fill(xvals, yvals, color=color, edgecolor='k', linewidth=font_size/12)

            if (color[0]*0.299*256 + color[1]*0.587*256 + color[2]*0.114*256) > 186:
                text_color = 'k'
            else:
                text_color = 'w'

            if plot_text == 'basis':
                if isinstance(q, MeasureQubit):
                    ax.text(coords[1], coords[0], f'{q.basis}', ha='center', va='center', color=text_color, fontsize=font_size)
            elif plot_text == 'idx':
                ax.text(coords[1], coords[0], f'{i}', ha='center', va='center', color=text_color, fontsize=font_size)
            elif plot_text == 'val' and qubit_vals is not None and np.isfinite(qubit_vals[i]):
                if val_fmt_fn is None:
                    exponent, rem = np.divmod(np.log10(qubit_vals[i]), 1)
                    if np.isfinite(exponent) and np.isfinite(rem):
                        ax.text(coords[1], coords[0], f'{10**rem:0.1f}e{int(exponent)}', ha='center', va='center', color=text_color, fontsize=font_size)
                else:
                    ax.text(coords[1], coords[0], val_fmt_fn(qubit_vals[i]), ha='center', va='center', color=text_color, fontsize=font_size)

        ax.set_xticks([])
        ax.set_yticks([])

        return ax, cbar

    def plot_connection_vals(
            self,
            connection_vals: dict[tuple[int, int], float],
            ax: plt.Axes | None = None,
        ):
        """Plot qubit connection values as a heatmap.
        
        Args:
            connection_vals: Array of connection values to plot.
            ax: Axes to plot on. If None, create new figure.
        """
        pass

class color:
    PURPLE = '\033[1;35;48m'
    CYAN = '\033[1;36;48m'
    BOLD = '\033[1;37;48m'
    BLUE = '\033[1;34;48m'
    GREEN = '\033[1;32;48m'
    YELLOW = '\033[1;33;48m'
    RED = '\033[1;31;48m'
    BLACK = '\033[1;30;48m'
    UNDERLINE = '\033[4;37;48m'
    END = '\033[1;37;0m'

    BLK = '\033[0;30m'
    RED = '\033[0;31m'
    GRN = '\033[0;32m'
    YEL = '\033[0;33m'
    BLU = '\033[0;34m'
    MAG = '\033[0;35m'
    CYN = '\033[0;36m'
    WHT = '\033[0;37m'

    BLKB = '\033[40m'
    REDB = '\033[41m'
    GRNB = '\033[42m'
    YELB = '\033[43m'
    BLUB = '\033[44m'
    MAGB = '\033[45m'
    CYNB = '\033[46m'
    WHTB = '\033[47m'

    BLKHB = '\033[0;100m'
    REDHB = '\033[0;101m'
    GRNHB = '\033[0;102m'
    YELHB = '\033[0;103m'
    BLUHB = '\033[0;104m'
    MAGHB = '\033[0;105m'
    CYNHB = '\033[0;106m'
    WHTHB = '\033[0;107m'