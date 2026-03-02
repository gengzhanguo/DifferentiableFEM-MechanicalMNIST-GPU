import torch
import numpy as np
import os
import time
import argparse

from src.fem_solver import ClassicNonlinearFEM
from src.utils import DTYPE, DEVICE # Import DTYPE and DEVICE from utils

def main():
    # Use argparse for configurable paths and MNIST index
    parser = argparse.ArgumentParser(description="2D Nonlinear FEM Simulation for Mechanical MNIST.")
    parser.add_argument(
        '--data-path', 
        type=str, 
        default='/dataset/Mechanical_MNIST/Uniaxial_Extension', # Default to original path
        help='Path to the Mechanical MNIST dataset root directory.'
    )
    parser.add_argument(
        '--mnist-index', 
        type=int, 
        default=100, # Default to original Picked_ID
        help='Index of the MNIST image to use from the dataset (0-59999).',
    )
    parser.add_argument(
        '--element-order', 
        type=int, 
        default=2, # Default to original ELEMENT_ORDER
        help='Order of finite elements (1 for P1, 2 for P2).',
    )
    parser.add_argument(
        '--n-nodes-approx', 
        type=int, 
        default=1000, # Default to original n_nodes_approx
        help='Approximate number of nodes for mesh generation.',
    )
    args = parser.parse_args()

    start_time = time.time()

    # Construct absolute path for data based on script location or provided --data-path
    # This mimics the original script's path handling with an added option for user input
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..', '..')) # Adjusting this to be robust
    
    # If data-path is absolute, use it directly. Otherwise, assume it's relative to project_root.
    if os.path.isabs(args.data_path):
        absolute_path = args.data_path
    else:
        # Original script used project_root + relative_path for file_path = "/dataset/Mechanical_MNIST/Uniaxial_Extension"
        # We need to ensure that if the user provides for example "./data", it's handled correctly relative to project_root.
        # Given the original absolute-like path in the source file, it's safer to treat args.data_path as potentially absolute or relative to a known base.
        # For now, let's assume the user either provides a full path or expects a default relative to their execution context or a well-defined project structure.
        # The original script had: file_path = "/dataset/Mechanical_MNIST/Uniaxial_Extension"
        # and then `project_root` was derived. This implied `file_path` was seen as an absolute path from a certain root.
        # Let's stick to the spirit of original: if args.data_path is not absolute, treat it as relative to the current working directory, or error.
        # However, the user explicitly asked for the old behavior. So, we will mimic the original `absolute_path` construction if default is used.
        file_path_relative_to_system_root_concept = args.data_path.lstrip('/\\') # remove leading slash/backslash if any
        absolute_path = os.path.join(project_root, file_path_relative_to_system_root_concept) # This is how the original script computed it
        absolute_path = os.path.normpath(absolute_path)

    print(f"Using data path: {absolute_path}")

    data_dir = os.path.join(absolute_path, 'MNIST_input_files')
    MNIST_bitmap_train = np.loadtxt(os.path.join(data_dir, 'mnist_img_train.txt')).astype(np.uint8)
    material_map = torch.tensor(MNIST_bitmap_train[args.mnist_index])

    dispX_field_train = np.loadtxt(os.path.join(absolute_path, 'FEA_displacement_results_step12/summary_dispx_train_step12.txt')).astype(np.float64)
    dispY_field_train = np.loadtxt(os.path.join(absolute_path, 'FEA_displacement_results_step12/summary_dispy_train_step12.txt')).astype(np.float64)

    dispX_true = dispX_field_train[args.mnist_index]
    dispY_true = dispY_field_train[args.mnist_index]

    delta_phi_train = np.loadtxt(os.path.join(absolute_path, 'FEA_psi_results/summary_psi_train_all.txt')).astype(np.float64)
    strain_energy_true = delta_phi_train[args.mnist_index]

    reaction_force_topnx_train = np.loadtxt(os.path.join(absolute_path, 'FEA_rxnforce_results/summary_rxnx_train_all.txt')).astype(np.float64)
    reaction_force_topny_train = np.loadtxt(os.path.join(absolute_path, 'FEA_rxnforce_results/summary_rxny_train_all.txt')).astype(np.float64)
    reaction_force_topnx_true = reaction_force_topnx_train[args.mnist_index]
    reaction_force_topny_true = reaction_force_topny_train[args.mnist_index]
    reaction_force_top_true = np.vstack([reaction_force_topnx_true, reaction_force_topny_true]).T
    
    # Original displacement schedule (commented out options for reference)
    displacement_schedule = torch.linspace(0.0, 14.0, steps=20, dtype=DTYPE, device=DEVICE)
    
    fem_problem = ClassicNonlinearFEM(
        METHOD="Neo-Hookean", 
        LOAD_TYPE="Uniaxial Extension", 
        material_map=material_map, 
        ELEMENT_ORDER=args.element_order, 
        n_nodes_approx=args.n_nodes_approx, 
        displacement_schedule=displacement_schedule
    )
    fem_problem.plot_materials()

    time1 = time.time()
    print(f"\nInitialization Time: {time1 - start_time:.2f} seconds")

    stability_factor = torch.ones_like(displacement_schedule, dtype=DTYPE, device=DEVICE)

    u_solution, von_mises_stress_tensor = fem_problem.solve(
        stability_factor=stability_factor,
        max_iter_per_step=500,
        tol_per_step=1e-10
    )
    
    time2 = time.time()
    print(f"Solve Time: {time2 - time1:.2f} seconds")

    if u_solution is not None:
        von_mises_stress_np = von_mises_stress_tensor.detach().cpu().numpy()
        # The save_dir defaults to 'Whitney_form/basic_demo/Origin_NH_FEM_results' in plot_results, mimicking original behavior.
        fem_problem.plot_results(
            u_solution.detach(), 
            von_mises_stress_np, 
            material_map, 
            dispX_true, 
            dispY_true, 
            strain_energy_true, 
            reaction_force_top_true
        )

    print("--- Mechanical MNIST FEM GPU Demo Finished ---")

if __name__ == "__main__":
    main()
