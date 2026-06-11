
"""
This script is responsible for generating training data by evaluating
random parameterisations of a quantum eigensolver implementation against a given Hamiltonian.

This is the code of Phase 1 - data augmentation logic.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
from kcl_util import save_to_binary, flatten, ham_to_vector, vec_to_fixed_size_vec, print_to_file

# Imports for ML part
import numpy as np
import stopit
import random
import copy

# Just mine the data from 1 Hamiltonian for getting a set of 1000 samples
# Timeout is set to 660  seconds
def miner(n_qubits, ham, repeats, timeout, file_X, file_Y, generator_caller, wrapper_caller, compress_caller):
    # Set precision high enough for this alg.
    np.set_printoptions(precision=17)

    # Initialise lists to store X and Y data
    X = [] # Random wrappers created from hamiltonians in the folder
    Y = [] # Energy level computed classically for each wrapper in X

    # Create a vector once, to be appended to x
    ham_vec = ham_to_vector(ham)

    # Make sure params are sensible
    repeats = max(repeats, 5)
    timeout = max(timeout, 20)

    # Compress to wrapper
    orig=str(len(ham.terms))
    compress_caller(ham)
    print_to_file(">>> compress to " + str(len(ham.terms)) + " from " + orig)

    # Add some random value with the wrapper
    for i in range(0, repeats):  # Loop from 1 to 1000
        print_to_file(">>>> Collecting data with qubits "+str(n_qubits) + " iteration " + str(i))
        previous_run_id = os.environ.get("RUN_ID")
        os.environ["RUN_ID"] = str(i)
        x_vec_params = generator_caller(i, n_qubits)
        y_vec=0

        try:
            # Call the wrapper
            with stopit.ThreadingTimeout(timeout) as context_manager:
                y_vec=wrapper_caller(x_vec_params, n_qubits, ham)

            # Did the code finish running in under 120 seconds?
            if context_manager.state == context_manager.TIMED_OUT:
                y_vec=3.0
                print_to_file("Y is 0 due to timeout")

            # Add to array - for later training
            x_vec = np.append(x_vec_params, ham_vec)
            X.append(x_vec)  # Parameters and ham as a vector
            Y.append(y_vec)  # E level classically
            print_to_file(">>>> Register data, X size of " + str(len(x_vec)) + " with energy level " + str(y_vec))

        except Exception as e:
                y_vec=0.0
                print(e)
                print_to_file("Y is 0 due to exception")
        finally:
            if previous_run_id is None:
                os.environ.pop("RUN_ID", None)
            else:
                os.environ["RUN_ID"] = previous_run_id

        print_to_file(">> End Collecting Data Samples, y:= " + str(y_vec))
        
        print_to_file(x_vec_params)
        # x_train = Wrapper(n_qubits, ham, False, True, 100, 0.001, False, 100, 10**5, 1e-6, 5, 128, 2, 0)
        # y_train = Energy level (computed classically) # We make a BIG assumption that the SVM will be able to generalise without retraining on 28 qubits
    # End of loop

    # End Mining - test we are not writing grabage.
    print_to_file(">> End Mining Phase. With " + str(len(X)) + " training data")
    if len(X) != len(Y):
        print("size of X: " + str(len(X)))
        print("size of Y: " + str(len(Y)))
        raise ValueError("Size of X and Y must be equal")

    # Write to file X and Y
    save_to_binary(file_X, X)
    save_to_binary(file_Y, Y)

    # Returns how many items really got out of the 1000 tries
    return len(X)
