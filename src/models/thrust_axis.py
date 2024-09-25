import numpy as np
from time import time
# algorithm taken from https://software.belle2.org/light-2408-savannah/doxygen/Thrust_8cc_source.html and adapted to python
class Thrust:
    @staticmethod
    def calculate_thrust(momenta):
        """
        Calculate thrust axis for a given list of momenta vectors.
        momenta: list of numpy arrays representing 3D momentum vectors.
        Returns: numpy array representing the thrust axis.
        """
        t_start = time()
        # STEP 1: Initialization of Variables
        thrust_axis = np.zeros(3)
        trial_axis = np.zeros(3)
        base_axis = np.zeros(3)
        sum_magnitude_mom = 0.0
        thrust = 0.0
        # STEP 2: Compute total magnitude Σ(||p_i||)
        for momentum in momenta:
            sum_magnitude_mom += np.linalg.norm(momentum)
        # STEP 3: For each momentum in momenta, use it as initial axis

        if len(momenta) > 100: # temporary
            # The 10 percent of the highest energy hits
            smaller_momenta = sorted(momenta, key=lambda x: np.linalg.norm(x))[:int(0.5 * len(momenta))]
        else:
            smaller_momenta = momenta
        for mom in smaller_momenta:
            # By convention, thrust axis in the same direction as Z axis
            trial_axis = mom if mom[2] >= 0 else -mom
            trial_axis = trial_axis / np.linalg.norm(trial_axis) if np.linalg.norm(trial_axis) != 0 else trial_axis
            while True:
                # STEP 4: Store previous trial axis as base axis, then reinitialize trial axis
                base_axis = trial_axis.copy()
                trial_axis = np.zeros(3)
                # Z-alignment of momenta and sum them to form new trial axis
                for momentum in momenta:
                    if np.dot(momentum, base_axis) >= 0:
                        trial_axis += momentum
                    else:
                        trial_axis -= momentum
                # STEP 5: Check condition ( p_i · trial_axis ) * ( p_i · base_axis ) < 0 for all p_i
                for momentum in momenta:
                    if np.dot(momentum, trial_axis) * np.dot(momentum, base_axis) < 0:
                        break
                else:
                    # If no break, exit while loop (condition met for all momenta)
                    break
            trial_mag = np.linalg.norm(trial_axis)
            if trial_mag != 0:
                trial_axis /= trial_mag
            # STEP 7: Compute thrust associated with selected trial axis
            trial_thrust = sum(abs(np.dot(momentum, trial_axis)) for momentum in momenta) / sum_magnitude_mom
            # STEP 8: Keep trial axis as thrust axis if it's better
            if trial_thrust > thrust:
                thrust = trial_thrust
                thrust_axis = trial_axis
        # STEP 10: Multiply normalized thrust axis by thrust
        thrust_axis *= thrust
        t_end = time()
        print("Calculating thrust took", t_end - t_start, "seconds")
        return thrust_axis

def hits_xyz_to_momenta(hits_xyz, hits_E):
    # barycenter is a weighted average of hits
    barycenter = np.average(hits_xyz, weights=hits_E, axis=0)
    # momenta is the vector from barycenter to each hit
    unit_vectors = [hit_xyz - barycenter for hit_xyz in hits_xyz]
    unit_vectors = np.stack(unit_vectors)
    unit_vectors /= np.linalg.norm(unit_vectors, axis=1)[:, np.newaxis]
    momenta = unit_vectors * hits_E[:, np.newaxis]
    return momenta
if __name__ == "__main__":
    # demo for debugging
    # Example usage
    momenta = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]  # Replace with actual vectors
    thrust_axis = Thrust.calculate_thrust(momenta)
    print("Thrust Axis:", thrust_axis)

    # Gaussian around (1, 0, 0) with variance 0.5 in the x-axis direction and 0.1 in the other 2
    #hits_xyz = np.random.normal(0.33, 0.1, (500, 3)) # 500 hits
    hits_x = np.random.normal(1, 0.5, 500)
    hits_y = np.random.normal(0, 0.1, 500)
    hits_z = np.random.normal(0, 0.1, 500)
    hits_xyz = np.stack([hits_x, hits_y, hits_z], axis=1)
    hits_E = np.random.uniform(0, 1, 500)
    momenta = hits_xyz_to_momenta(hits_xyz, hits_E)  # Relative to the energy-weighted barycenter of the hits
    thrust_axis = Thrust.calculate_thrust(momenta)
    print("Thrust Axis:", thrust_axis)

