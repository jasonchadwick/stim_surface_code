import stim
from stim_surface_code.patch import SurfaceCodePatch, DataQubit, MeasureQubit
    
class StabilityPatch(SurfaceCodePatch):
    """Surface code patch that performs the memory experiment. See
    https://quantum-journal.org/papers/q-2022-08-24-786/ for details.
    """
    def __init__(
            self, 
            dx: int,
            dz: int,
            dm: int,
            observable_basis: str = 'Z',
            *args,
            **kwargs,
        ) -> None:
        """Initialize.
        
        Args:
            dx: X distance of code (~ number of rows of qubits).
            dz: Z distance of code (~ number of columns of qubits).
            dm: Temporal distance of code (= number of rounds of measurement).
            boundary_basis: Basis of boundary stabilizers. Must be 'X' or 'Z'.
            args: Arguments to pass to SurfaceCodePatch.__init__.
            kwargs: Keyword arguments to pass to SurfaceCodePatch.__init__.
        """
        self.observable_basis = observable_basis
        self.boundary_basis = 'X' if observable_basis == 'Z' else 'Z'

        super().__init__(dx, dz, dm, *args, **kwargs)

    def place_ancilla(self) -> None:
        """Place ancilla (non-data) qubits in the patch. Must be run *after*
        place_data.
        """
        # number of qubits already placed (= index of next qubit)
        q_count = len(self.data)

        self.observable_ancilla: list[MeasureQubit] = []
        self.boundary_basis_ancilla: list[MeasureQubit] = []
        for row in range(self.dx+1):
            for col in range(self.dz+1):
                if (row + col) % 2 == 1 and not ((row == self.dx and col == 0) or (row == 0 and col == self.dz)): # Boundary basis
                    coords = (2*row, 2*col)
                    data_qubits = self._get_neighboring_data_qubits(coords)
                    if all(q is None for q in data_qubits):
                        continue
                    measure_q = MeasureQubit(q_count, coords, data_qubits, self.boundary_basis)
                    self.device[coords[0]][coords[1]] = measure_q
                    self.boundary_basis_ancilla.append(measure_q)
                    q_count += 1
                elif (row + col) % 2 == 0 and row != 0 and row != self.dx and col != 0 and col != self.dz: # Z basis
                    coords = (2*row, 2*col)
                    data_qubits = self._get_neighboring_data_qubits(coords)
                    if all(q is None for q in data_qubits):
                        continue
                    measure_q = MeasureQubit(q_count, coords, data_qubits, self.observable_basis)
                    self.device[coords[0]][coords[1]] = measure_q
                    self.observable_ancilla.append(measure_q)
                    q_count += 1
        self.z_ancilla = (
            self.observable_ancilla if self.observable_basis == 'Z'
            else self.boundary_basis_ancilla
        )
        self.x_ancilla = (
            self.observable_ancilla if self.observable_basis == 'X'
            else self.boundary_basis_ancilla
        )
    
    def get_stim(self) -> stim.Circuit:
        """Generate Stim code performing a stability experiment in desired basis.
        
        Args:
            observable_basis: Basis to prepare and measure in. Must be 'X', 'Y'
                or 'Z'. 
            ideal_init_and_meas: If True, perform ideal initialization and
                measurement instead of gate-based. Required if basis is 'Y'.
        
        Returns:
            Stim circuit implementing the logical stability experiment.
        """
        assert self.error_vals_initialized

        self.meas_record: list[dict[int, int]] = []
        
        observable_ancilla, boundary_basis_ancilla = (
            (self.x_ancilla, self.z_ancilla) if self.observable_basis == 'X'
            else (self.z_ancilla, self.x_ancilla)
        )

        circ = stim.Circuit()

        # Coords
        for qubit in self.all_qubits:
            circ.append('QUBIT_COORDS', qubit.idx, qubit.coords)

        # Syndrome rounds
        self.syndrome_round(
            circ, 
            deterministic_detectors=[q.idx for q in self.observable_ancilla], 
            inactive_detectors=[q.idx for q in self.boundary_basis_ancilla],
        )
        circ.append(stim.CircuitRepeatBlock(self.dm - 1, self.syndrome_round(stim.Circuit())))

        # Measure in observable basis
        if self.observable_basis == 'X':
            self.apply_1gate(circ, 'H', [q.idx for q in self.data])
        self.apply_meas(circ, [q.idx for q in self.data])

        # Check consistency of data qubit measurements with last stabilizer measurement
        for measure in observable_ancilla:
            data_rec = [self.get_meas_rec(-1, data.idx) for data in measure.data_qubits if data is not None]
            circ.append('DETECTOR', data_rec + [self.get_meas_rec(-2, measure.idx)], measure.coords + (0,))

        # Observable is parity of all stabilizers matching boundary basis
        circ.append("OBSERVABLE_INCLUDE", [self.get_meas_rec(-2, ancilla.idx) for ancilla in self.boundary_basis_ancilla], 0)
        
        return circ
    