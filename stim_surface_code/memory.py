"""TODO
"""
import stim
from stim_surface_code.patch import SurfaceCodePatch, DataQubit, MeasureQubit

class MemoryPatch(SurfaceCodePatch):
    """Surface code patch that performs the memory experiment."""
    def __init__(
            self, 
            dx: int,
            dz: int,
            dm: int,
            *args,
            **kwargs,
        ) -> None:
        """Initialize.
        
        Args:
            dx: X distance of code (~ number of rows of qubits).
            dz: Z distance of code (~ number of columns of qubits).
            dm: Temporal distance of code (= number of rounds of measurement).
            args: Arguments to pass to SurfaceCodePatch.__init__.
            kwargs: Keyword arguments to pass to SurfaceCodePatch.__init__.
        """
        super().__init__(dx, dz, dm, *args, **kwargs)

    def get_stim(
            self, 
            observable_basis: str = 'Z', 
            ideal_init_and_meas: bool = True,
        ) -> stim.Circuit:
        """Generate Stim code performing a memory experiment in desired basis.
        
        Args:
            observable_basis: Basis to prepare and measure in. Must be 'X', 'Y'
                or 'Z'. 
            ideal_init_and_meas: If True, perform ideal initialization and
                measurement instead of gate-based. Required if basis is 'Y'.
        
        Returns:
            Stim circuit implementing the logical memory experiment.
        """
        assert self.error_vals_initialized
        assert observable_basis != 'Y' or ideal_init_and_meas

        self.meas_record: list[dict[int, int]] = []

        observable_ancilla = (
            self.z_ancilla if observable_basis == 'Z' else self.x_ancilla
        )
        circ = stim.Circuit()

        # Coords
        for qubit in self.all_qubits:
            circ.append('QUBIT_COORDS', qubit.idx, qubit.coords)
        
        if ideal_init_and_meas:
            self.initialize_ideal(circ, {'X':'+', 'Y':'i', 'Z':'0'}[observable_basis])
        else:
            self.apply_reset(circ, [q.idx for q in self.all_qubits])
            if observable_basis == 'X':
                self.apply_1gate(circ, 'H', [q.idx for q in self.data])

        # Syndrome rounds
        self.syndrome_round(
            circ,
            deterministic_detectors=[a.idx for a in observable_ancilla],
            inactive_detectors=[a.idx for a in self.ancilla 
                                if a not in observable_ancilla],
        )
        circ.append(stim.CircuitRepeatBlock(self.dm - 1, self.syndrome_round(stim.Circuit())))

        # measurement
        if ideal_init_and_meas:
            self.meas_ideal(circ, 'M' if observable_basis == 'Z' else 'MX', [data.idx for data in self.data])
        else:
            # Measure in observable basis
            if observable_basis == 'X':
                self.apply_1gate(circ, 'H', [q.idx for q in self.data])
            self.apply_meas(circ, [q.idx for q in self.data])
            
        # Check consistency of data qubit measurements with last stabilizer measurement
        for measure in observable_ancilla:
            data_rec = [self.get_meas_rec(-1, data.idx) for data in measure.data_qubits if data is not None]
            circ.append('DETECTOR', data_rec + [self.get_meas_rec(-2, measure.idx)], measure.coords + (0,))

        # Observable is observables of qubits along logical operator
        circ.append(
            'OBSERVABLE_INCLUDE', 
            [self.get_meas_rec(-1, data.idx) 
                for data in (
                    self.logical_z_qubits if observable_basis == 'Z' 
                    else self.logical_x_qubits)], 
            0)

        return circ