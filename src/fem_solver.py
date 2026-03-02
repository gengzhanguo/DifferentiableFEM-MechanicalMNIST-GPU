import torch
import numpy as np
import skfem
from skfem import Functional
import matplotlib.pyplot as plt
# plt.switch_backend('TkAgg')
import matplotlib.tri as mtri
import skfem.visuals.matplotlib as femplot
import torch.func
from scipy.sparse.linalg import spsolve
import pygmsh
import tempfile
import os
import scipy
from typing import List, Tuple, Optional
import matplotlib as mpl

# Import global configs and utility functions
from .utils import DTYPE, DEVICE, sparse_coo_to_csc

mpl.rcParams.update({
    "font.size": 10,            # Default text size
    "axes.titlesize": 10,      # Axes title
    "axes.labelsize": 10,      # Axes labels
    "xtick.labelsize": 8,     # X tick labels
    "ytick.labelsize": 8,     # Y tick labels
    "legend.fontsize": 8,     # Legend text
    "figure.titlesize": 10,    # Figure title
})

print(f"PyTorch Version: {torch.__version__}")
print(f"Using device: {DEVICE}")

class ClassicNonlinearFEM:
    """
    Solves a 2D nonlinear hyperelasticity problem with heterogeneous materials
    using a fully differentiable Finite Element Method in PyTorch.
    """
    def __init__(self, METHOD: str, LOAD_TYPE: str, material_map: torch.Tensor, ELEMENT_ORDER: int = 1, n_nodes_approx: int = 400, displacement_schedule: Optional[torch.Tensor] = None):
        """
        Initializes the FEM problem by generating a mesh, defining basis functions,
        assigning material properties from the map, and setting up boundary conditions.

        Args:
            material_map (torch.Tensor): A 1D tensor of size 784 representing the material layout.
            n_nodes_approx (int): The approximate number of nodes for mesh generation.
        """

        self.METHOD = METHOD # "St.Venant-Kirchhoff" or "Neo-Hookean"
        self.LOAD_TYPE = LOAD_TYPE # "Uniaxial Extension", "Pure Shear", "Equibiaxial Extension", or "Uniaxial Compression"
        print(f"--- Using Backbone Physical Material Model: {self.METHOD} ---")
        print(f"--- Using Load Type: {self.LOAD_TYPE} ---") 
        print("--- Initializing Mesh and FE Space using pygmsh ---")
        
        self.DOMAIN_SIZE = 28.0

        # Material properties will now be defined by the material_map
        self.E_MODULUS_STIFF = 100.0  # Young's Modulus for stiff material (white pixels)
        self.E_MODULUS_SOFT = 1.0     # Young's Modulus for soft material (black pixels)
        self.NU_POISSON = 0.3         # Poisson's Ratio (kept constant)
        
        # --- Mesh Generation ---
        n_points_per_side = int(np.sqrt(n_nodes_approx))
        mesh_size = self.DOMAIN_SIZE / (n_points_per_side - 1) if n_points_per_side > 1 else self.DOMAIN_SIZE
        
        with pygmsh.geo.Geometry() as geom:
            geom.add_rectangle(0.0, self.DOMAIN_SIZE, 0.0, self.DOMAIN_SIZE, 0.0, mesh_size=mesh_size)
            pygmsh_mesh = geom.generate_mesh()
        
        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
            tmp_filename = tmp.name
            pygmsh_mesh.write(tmp_filename, file_format="gmsh22")
        
        mesh = skfem.MeshTri.load(tmp_filename)
        os.remove(tmp_filename)

        mesh = mesh.with_boundaries({
            "bottom": lambda x: x[1] < 1e-6,
            "top": lambda x: x[1] > self.DOMAIN_SIZE - 1e-6,
            "left": lambda x: x[0] < 1e-6,
            "right": lambda x: x[0] > self.DOMAIN_SIZE - 1e-6,
        })

        self.mesh = mesh # Store mesh for later use
        
        self.ELEMENT_ORDER = ELEMENT_ORDER
        element = skfem.ElementTriP1() if ELEMENT_ORDER == 1 else skfem.ElementTriP2()
        self.basis = skfem.Basis(mesh, element)
        
        self.nodes = torch.tensor(self.basis.doflocs.T, dtype=DTYPE, device=DEVICE)

        self.elements = torch.tensor(self.basis.element_dofs.T, dtype=torch.long, device=DEVICE)

        self.n_nodes = self.basis.N
        self.n_elements = self.elements.shape[0]
        self.n_dofs = self.n_nodes * 2
        
        print(f"Mesh Initialized: {self.n_nodes} nodes, {self.n_elements} elements, {self.n_dofs} DOFs")
        
        self.I = torch.eye(2, dtype=DTYPE, device=DEVICE)

        self._assign_heterogeneous_materials(material_map, self.basis)

        print("Pre-computing basis function gradients...")
        quad_points, quad_weights_np = self.basis.quadrature
        self.quad_weights = torch.tensor(quad_weights_np, dtype=DTYPE, device=DEVICE)
        detJ_np = np.abs(self.basis.mapping.detDF(quad_points))
        self.detJ = torch.tensor(detJ_np, dtype=DTYPE, device=DEVICE)

        n_quad_per_elem = self.quad_weights.shape[0]
        n_basis_funcs = self.elements.shape[1]

        all_global_grads_np = np.zeros((self.n_nodes, self.n_elements, n_quad_per_elem, 2))
        for i in range(self.n_nodes):
            x_basis = np.zeros(self.n_nodes)
            x_basis[i] = 1.0
            interpolant = self.basis.interpolate(x_basis)
            grad_vals = interpolant.grad.transpose(1, 2, 0)
            all_global_grads_np[i] = grad_vals
            
        basis_grads_np = np.zeros((self.n_elements, n_basis_funcs, n_quad_per_elem, 2))
        elements_np = self.elements.cpu().numpy()
        for e in range(self.n_elements):
            for k in range(n_basis_funcs):
                global_node_idx = elements_np[e, k]
                basis_grads_np[e, k] = all_global_grads_np[global_node_idx, e, :, :]
        
        self.basis_grads = torch.tensor(basis_grads_np, dtype=DTYPE, device=DEVICE)

        self.displacement_schedule = displacement_schedule
        self.strain_energy_list = []
        self.reaction_force_list = []

        self._set_default_boundary_conditions()

    def _set_default_boundary_conditions(self):
        """Sets the standard stretching boundary conditions."""
        self.dirichlet_mask = torch.zeros(self.n_dofs, dtype=torch.bool, device=DEVICE)
        self.dirichlet_values = torch.zeros(self.n_dofs, dtype=DTYPE, device=DEVICE)
        
        self.bottom_nodes = self.basis.get_dofs("bottom").all()
        self.top_nodes = self.basis.get_dofs("top").all()

        def on_left_edge(x):
            return np.isclose(x[0], 0.0)
        def on_right_edge(x):
            return np.isclose(x[0], self.DOMAIN_SIZE)
        self.left_nodes = self.basis.get_dofs(on_left_edge).all()
        self.right_nodes = self.basis.get_dofs(on_right_edge).all()


        if self.LOAD_TYPE == "Uniaxial Extension" or self.LOAD_TYPE == "Uniaxial Compression" or self.LOAD_TYPE == "Pure Shear":
            # Set Dirichlet boundary conditions, mute for free boundaries in specific edges.
            self.dirichlet_mask[self.bottom_nodes * 2] = True # X displacement at bottom
            self.dirichlet_values[self.bottom_nodes * 2] = 0 # X displacement = 0 at bottom (will change in solver)
            self.dirichlet_mask[self.bottom_nodes * 2 + 1] = True # Y displacement at bottom
            self.dirichlet_values[self.bottom_nodes * 2 + 1] = 0 # Y displacement = 0 at bottom (will change in solver)
            
            self.dirichlet_mask[self.top_nodes * 2] = True # X displacement at top
            self.dirichlet_values[self.top_nodes * 2] = 0 # X displacement = 0 at top
            self.dirichlet_mask[self.top_nodes * 2 + 1] = True # Y displacement at top
            self.dirichlet_values[self.top_nodes * 2 + 1] = 0 # Y displacement = 0 at top (will change in solver)

        elif self.LOAD_TYPE == "Equibiaxial Extension":
            self.dirichlet_mask[self.bottom_nodes * 2 + 1] = True
            self.dirichlet_values[self.bottom_nodes * 2 + 1] = 0

            self.dirichlet_mask[self.top_nodes * 2 + 1] = True
            self.dirichlet_values[self.top_nodes * 2 + 1] = 0
            
            self.dirichlet_mask[self.left_nodes * 2] = True
            self.dirichlet_values[self.left_nodes * 2] = 0

            self.dirichlet_mask[self.right_nodes * 2] = True
            self.dirichlet_values[self.right_nodes * 2] = 0
        else:
            raise ValueError(f"Unknown load type: {self.LOAD_TYPE}")
        
    def _assign_heterogeneous_materials(self, material_map: torch.Tensor, basis: skfem.Basis):
        """Maps the input pixel data to material properties at each quadrature point."""
        print("--- Assigning heterogeneous material properties to quadrature points ---")

        youngs_modulus_map = material_map.to(DTYPE).to(DEVICE) / 255.0 * (self.E_MODULUS_STIFF - self.E_MODULUS_SOFT) + self.E_MODULUS_SOFT

        quad_points_local, _ = basis.quadrature
        self.quad_points_global_np = basis.mapping.F(quad_points_local)
        quad_points_flat = torch.tensor(self.quad_points_global_np.transpose(1, 2, 0).reshape(-1, 2), dtype=DTYPE, device=DEVICE)

        img_size = 28
        pixel_width = self.DOMAIN_SIZE / img_size
        x_coords = torch.arange(img_size, device=DEVICE) * pixel_width
        y_coords = torch.arange(img_size, device=DEVICE) * pixel_width
        
        yv_min, xv_min = torch.meshgrid(torch.flip(y_coords, [0]), x_coords, indexing='ij')
        
        pixel_boxes = torch.stack([
            xv_min.flatten(),
            xv_min.flatten() + pixel_width,
            yv_min.flatten(),
            yv_min.flatten() + pixel_width
        ], dim=1)

        quad_pts_expanded = quad_points_flat.unsqueeze(1)
        pixel_boxes_expanded = pixel_boxes.unsqueeze(0)
        
        mask_x = (quad_pts_expanded[..., 0] >= pixel_boxes_expanded[..., 0]) & (quad_pts_expanded[..., 0] < pixel_boxes_expanded[..., 1])
        mask_y = (quad_pts_expanded[..., 1] >= pixel_boxes_expanded[..., 2]) & (quad_pts_expanded[..., 1] < pixel_boxes_expanded[..., 3])
        pixel_mask = mask_x & mask_y

        pixel_indices = torch.argmax(pixel_mask.to(torch.int8), dim=1)
        E_quad_flat = youngs_modulus_map[pixel_indices]

        no_match_mask = ~pixel_mask.any(dim=1)
        if no_match_mask.any():
            E_quad_flat[no_match_mask] = self.E_MODULUS_SOFT

        lmbda_flat = E_quad_flat * self.NU_POISSON / ((1 + self.NU_POISSON) * (1 - 2 * self.NU_POISSON))
        mu_flat = E_quad_flat / (2 * (1 + self.NU_POISSON))

        n_quad_per_elem = quad_points_local.shape[1]
        self.lmbda_quad = lmbda_flat.reshape(self.n_elements, n_quad_per_elem, 1, 1)
        self.mu_quad = mu_flat.reshape(self.n_elements, n_quad_per_elem, 1, 1)
        print("Material assignment complete.")

    def compute_internal_forces(self, u_flat: torch.Tensor) -> torch.Tensor:
        """
        Computes the internal force vector (the integral part of the residual).
        Inputs:
            u_flat (torch.Tensor): Flattened displacement vector of size (n_dofs,).
        Internal:
            self.elements: (n_elements, n_basis_funcs)
            self.basis_grads: (n_elements, n_basis_funcs, n_quad_per_elem, 2)
            self.lmbda_quad: (n_elements, n_quad_per_elem, 1, 1)
            self.mu_quad: (n_elements, n_quad_per_elem, 1, 1)
        Returns:
            internal_force_flat (torch.Tensor): Flattened internal force vector of size (n_dofs,).
        
        e: n_elements
        k: n_basis_funcs
        q: n_quad_per_elem
        i,d: spatial dimensions (2D)
        """
        u = u_flat.reshape(self.n_nodes, 2) # Reshape to (n_nodes, 2)
        # self.elements: (n_elements, n_basis_funcs)
        u_elements = u[self.elements] # (n_elements, n_basis_funcs, 2)

        # self.basis_grads: (n_elements, n_basis_funcs, n_quad_per_elem, 2)
        grad_u = torch.einsum('eki,ekqd->eqid', u_elements, self.basis_grads) # (n_elements, n_quad_per_elem, 2, 2)
        F = self.I[None, None, :, :] + grad_u # (n_elements, n_quad_per_elem, 2, 2)

        if self.METHOD == "Neo-Hookean":
            tr_grad_u = grad_u.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) # (n_elements, n_quad_per_elem)
            det_grad_u = grad_u[:, :, 0, 0] * grad_u[:, :, 1, 1] - grad_u[:, :, 0, 1] * grad_u[:, :, 1, 0] # (n_elements, n_quad_per_elem)

            det_F = 1.0 + tr_grad_u + det_grad_u # (n_elements, n_quad_per_elem)
            det_F_expanded = det_F.unsqueeze(-1).unsqueeze(-1) # (n_elements, n_quad_per_elem, 1, 1)

            inv_F_T = torch.zeros_like(F)
            inv_F_T[:, :, 0, 0] = (1.0 + grad_u[:, :, 1, 1]) / det_F
            inv_F_T[:, :, 0, 1] = (-grad_u[:, :, 1, 0]) / det_F
            inv_F_T[:, :, 1, 0] = (-grad_u[:, :, 0, 1]) / det_F
            inv_F_T[:, :, 1, 1] = (1.0 + grad_u[:, :, 0, 0]) / det_F


            P = F * self.mu_quad + (0.5 * (det_F_expanded**2 - 1) * self.lmbda_quad - self.mu_quad) * inv_F_T # (n_elements, n_quad_per_elem, 2, 2)

        elif self.METHOD == "St.Venant-Kirchhoff":
            E = 0.5 * (F.transpose(-2, -1) @ F - self.I)
            tr_E = E.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
            
            S = self.lmbda_quad * tr_E * self.I + 2 * self.mu_quad * E
            P = F @ S

        else:
            raise ValueError(f"Unknown material model: {self.METHOD}")

        internal_forces_element = torch.einsum('eqid,ekqd,eq,q->eki', P, self.basis_grads, self.detJ, self.quad_weights)

        internal_force_flat = torch.zeros_like(u_flat)
        element_dofs = (self.elements * 2).unsqueeze(2) + torch.arange(2, device=DEVICE)
        internal_force_flat.scatter_add_(0, element_dofs.flatten(), internal_forces_element.flatten())
        return internal_force_flat

    def compute_detF(self, u_flat: torch.Tensor) -> torch.Tensor:
        u = u_flat.reshape(self.n_nodes, 2) # Reshape to (n_nodes, 2)
        u_elements = u[self.elements] # (n_elements, n_basis_funcs, 2)
        grad_u = torch.einsum('eki,ekqd->eqid', u_elements, self.basis_grads) # (n_elements, n_quad_per_elem, 2, 2)
        F = self.I[None, None, :, :] + grad_u # (n_elements, n_quad_per_elem, 2, 2)
        tr_grad_u = grad_u.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) # (n_elements, n_quad_per_elem)
        det_grad_u = grad_u[:, :, 0, 0] * grad_u[:, :, 1, 1] - grad_u[:, :, 0, 1] * grad_u[:, :, 1, 0] # (n_elements, n_quad_per_elem)
        det_F = 1.0 + tr_grad_u + det_grad_u # (n_elements, n_quad_per_elem)
        return det_F

    def compute_residual(self, u_flat: torch.Tensor) -> torch.Tensor:
        """Computes the global residual vector."""
        return self.compute_internal_forces(u_flat)

    def _solve_newton_raphson(self, u_init: Optional[torch.Tensor] = None, stability_factor: float = 1.0, max_iter: int = 50, tol: float = 1e-9) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Solves the nonlinear system using a Newton-Raphson method.
        Can start from an optional initial guess u_init.
        """
        print("\n--- Starting Newton-Raphson Solver (Differentiable, PyTorch CG) ---")
        
        if u_init is not None:
            u = u_init.clone()
            # Ensure the initial guess still respects the boundary conditions
            u[self.dirichlet_mask] = self.dirichlet_values[self.dirichlet_mask]
        else:
            u = torch.zeros(self.n_dofs, dtype=DTYPE, device=DEVICE)
            u[self.dirichlet_mask] = self.dirichlet_values[self.dirichlet_mask]


        def get_element_jacobian(u_element_flat, basis_grads_e, lmbda_e, mu_e, detJ_e):
            """Computes the 6x6 Jacobian for one element with its material properties."""
            def element_residual_closure(u_e_flat):
                n_basis_funcs_e = basis_grads_e.shape[0]
                u_e_reshaped = u_e_flat.reshape(n_basis_funcs_e, 2)
                grad_u_e = torch.einsum('ki,kqd->qid', u_e_reshaped, basis_grads_e)
                F_e = self.I[None, :, :] + grad_u_e
                if self.METHOD == "Neo-Hookean":
                    tr_grad_u_e = grad_u_e.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
                    det_grad_u_e = grad_u_e[:, 0, 0] * grad_u_e[:, 1, 1] - grad_u_e[:, 0, 1] * grad_u_e[:, 1, 0]
                    det_F_e = 1.0 + tr_grad_u_e + det_grad_u_e
                    det_F_e_expanded = det_F_e.unsqueeze(-1).unsqueeze(-1)
                    inv_F_T_e = torch.zeros_like(F_e)
                    inv_F_T_e[:, 0, 0] = (1.0 + grad_u_e[:, 1, 1]) / det_F_e
                    inv_F_T_e[:, 0, 1] = (-grad_u_e[:, 1, 0]) / det_F_e
                    inv_F_T_e[:, 1, 0] = (-grad_u_e[:, 0, 1]) / det_F_e
                    inv_F_T_e[:, 1, 1] = (1.0 + grad_u_e[:, 0, 0]) / det_F_e
                    P_e = F_e * mu_e + (0.5 * (det_F_e_expanded**2 - 1) * lmbda_e - mu_e) * inv_F_T_e
                elif self.METHOD == "St.Venant-Kirchhoff":
                    E_e = 0.5 * (F_e.transpose(-2, -1) @ F_e - self.I)
                    tr_E_e = E_e.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
                    S_e = lmbda_e * tr_E_e * self.I + 2 * mu_e * E_e
                    P_e = F_e @ S_e
                else:
                    raise ValueError(f"Unknown material model: {self.METHOD}")
                weights_e = detJ_e * self.quad_weights # (q) -> (q)
                res_e = torch.einsum('qid,kqd,q->ki', P_e, basis_grads_e, weights_e)
                return res_e.flatten()
            
            return torch.func.jacrev(element_residual_closure)(u_element_flat)

        vectorized_jac_fn = torch.vmap(get_element_jacobian, in_dims=(0, 0, 0, 0, 0))

        n_dofs_per_element = self.elements.shape[1] * 2
        global_dof_indices_elements = (self.elements * 2).unsqueeze(2) + torch.arange(2, device=DEVICE)
        global_dof_indices_elements = global_dof_indices_elements.reshape(self.n_elements, n_dofs_per_element)
        
        rows, cols = torch.meshgrid(torch.arange(n_dofs_per_element), torch.arange(n_dofs_per_element), indexing='ij')
        rows_local, cols_local = rows.flatten(), cols.flatten()
        rows_global = global_dof_indices_elements[:, rows_local]
        cols_global = global_dof_indices_elements[:, cols_local]
        
        for i in range(max_iter):
            residual = self.compute_residual(u)
            res_norm = torch.linalg.norm(residual[~self.dirichlet_mask])
            print(f"Iteration {i+1}: Calculating Jacobians... (Residual norm = {res_norm:.4e})")

            if res_norm < tol and i > 0:
                print(f"Residual norm {res_norm:.4e} is below tolerance {tol}. Converged!")
                break
                
            u_elements_flat = u.reshape(self.n_nodes, 2)[self.elements].reshape(self.n_elements, -1)
            all_K_e = vectorized_jac_fn(u_elements_flat, self.basis_grads, self.lmbda_quad, self.mu_quad, self.detJ)

            vals_tensor = all_K_e.flatten()
            rows_tensor, cols_tensor = rows_global.flatten(), cols_global.flatten()
            
            jacobian_sparse_coo = torch.sparse_coo_tensor(
                indices=torch.stack([rows_tensor, cols_tensor]),
                values=vals_tensor,
                size=(self.n_dofs, self.n_dofs)
            ).coalesce()

            K_mod = jacobian_sparse_coo.clone()
            rhs = -residual
            rhs[self.dirichlet_mask] = 0.0

            bc_indices = self.dirichlet_mask.nonzero().flatten()
            mask = torch.ones(K_mod.values().shape[0], dtype=torch.bool, device=DEVICE)
            row_indices = K_mod.indices()[0]
            col_indices = K_mod.indices()[1]
            mask &= ~torch.isin(row_indices, bc_indices)
            mask &= ~torch.isin(col_indices, bc_indices)

            new_indices = K_mod.indices()[:, mask]
            new_values = K_mod.values()[mask]
            
            diag_indices = torch.stack([bc_indices, bc_indices])
            diag_values = torch.ones_like(bc_indices, dtype=DTYPE)
            
            final_indices = torch.cat([new_indices, diag_indices], dim=1)
            final_values = torch.cat([new_values, diag_values], dim=0)
            
            K_final = torch.sparse_coo_tensor(final_indices, final_values, (self.n_dofs, self.n_dofs)).coalesce()

            print(f"Iteration {i+1}: Solving sparse linear system...")
            K_final_scipy = K_final.cpu()
            K_final_scipy = sparse_coo_to_csc(
                K_final # Pass the sparse_coo_tensor directly
            )
            rhs_np = rhs.cpu().detach().numpy()
            
            try:
                du_np = spsolve(K_final_scipy, rhs_np)
            except Exception as e:
                print(f"!!! Direct solver 'spsolve' FAILED: {e} !!!")
                print("This might indicate a singular matrix (e.g., free-floating parts).")
                return None 
            du = torch.tensor(du_np, dtype=DTYPE, device=DEVICE)
            eta = stability_factor
            u_old = u.clone()
            for ls_iter in range(10):
                u = u_old + eta * du
                u[self.dirichlet_mask] = self.dirichlet_values[self.dirichlet_mask]
                
                new_res_norm = torch.linalg.norm(self.compute_residual(u)[~self.dirichlet_mask])
                
                if torch.any(self.compute_detF(u) <= 1e-6):
                    eta *= 0.5
                    continue

                if new_res_norm < res_norm:
                    break
                else:
                    eta *= 0.5 

            norm_du = torch.linalg.norm(du)
            print(f"Iteration {i+1}: Displacement update norm |du| = {norm_du:.4e}, with stability factor = {eta} for method '{self.METHOD}' and load type '{self.LOAD_TYPE}'.")
            if res_norm < tol:
                print(f"Residual norm {res_norm:.4e} is below tolerance {tol}. Converged!")
                break
            if norm_du < tol:
                print(f"\nSolver converged in {i+1} iterations!")
                break
        else:
            print("\nWarning: Newton's method did not converge within the maximum number of iterations.")
        
        u_reshaped = u.reshape(self.n_nodes, 2)
        return u_reshaped
    
    def solve(self, 
              stability_factor: List[float],
              max_iter_per_step: int = 50, 
              tol_per_step: float = 1e-9
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Solves the nonlinear system using incremental loading.
        
        Args:
            displacement_schedule (Optional[torch.Tensor]): 
                A 1D tensor of *top* displacement values to apply.
                If None, uses the default schedule from the user.
            max_iter_per_step (int): Max Newton iterations for *each* load step.
            tol_per_step (float): Convergence tolerance for *each* load step.
        
        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
                - The final displacement solution (N, 2 tensor)
                - The final Von Mises stress (N, tensor)
                - (None, None) if convergence fails.
        """
        
        if self.displacement_schedule is None:
            print("Using default displacement schedule.")
            self.displacement_schedule = torch.tensor(
                [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0], # 
                dtype=DTYPE, device=DEVICE
            )
        if stability_factor is None:
            stability_factor = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        
        u_solution_step_reshaped = None  # Stores the (N, 2) solution from the previous step
        
        total_steps = len(self.displacement_schedule)
        print(f"\n--- Starting Incremental Loading ({total_steps} steps) ---")

        for i, current_top_disp in enumerate(self.displacement_schedule):
            
            print(f"\n--- Load Step {i+1}/{total_steps} (Top Displacement = {current_top_disp:.4f}) ---")
            
            # 1. Update the boundary conditions for this step
            if self.LOAD_TYPE == "Uniaxial Extension":
                self.dirichlet_values[self.top_nodes * 2 + 1] = current_top_disp.item()
            if self.LOAD_TYPE == "Uniaxial Compression":
                self.dirichlet_values[self.top_nodes * 2 + 1] = -current_top_disp.item()
            if self.LOAD_TYPE == "Pure Shear":
                self.dirichlet_values[self.top_nodes * 2] = current_top_disp.item()
            if self.LOAD_TYPE == "Equibiaxial Extension":
                self.dirichlet_values[self.bottom_nodes * 2 + 1] = -current_top_disp.item()
                self.dirichlet_values[self.top_nodes * 2 + 1] = current_top_disp.item()
                self.dirichlet_values[self.left_nodes * 2] = -current_top_disp.item()
                self.dirichlet_values[self.right_nodes * 2] = current_top_disp.item()
            # 2. Prepare u_init (flat vector)
            u_init_flat = u_solution_step_reshaped.flatten() if u_solution_step_reshaped is not None else None
            
            # 3. Solve this single step
            u_solution_step_reshaped = self._solve_newton_raphson(
                u_init=u_init_flat, 
                stability_factor=stability_factor[i],
                max_iter=max_iter_per_step,
                tol=tol_per_step
            )
            
            # 4. Check for convergence
            if u_solution_step_reshaped is None:
                print(f"!!! Solver FAILED at load step {i+1} (Displacement = {current_top_disp:.4f}) !!!")
                return None, None # Return failure
            
            # 5. Store strain energy for this step
            self.strain_energy_list.append(self.compute_strain_energy(u_solution_step_reshaped))
            self.reaction_force_list.append(self.compute_reaction_forces(u_solution_step_reshaped))
            
        # --- Incremental Loading Finished ---
        if u_solution_step_reshaped is not None:
            print("\nIncremental loading completed successfully.")
            # Compute final stress
            von_mises_stress_tensor = self.compute_von_mises_stress(u_solution_step_reshaped)
            self.strain_energy_list = torch.stack(self.strain_energy_list)

            return u_solution_step_reshaped, von_mises_stress_tensor
        else:
            print("\nSolver failed during the incremental loading process.")
            return None, None
        
    def compute_von_mises_stress(self, u_solved_reshaped: torch.Tensor) -> torch.Tensor:
        """Computes nodal Von Mises stress using heterogeneous properties."""
        u_elements = u_solved_reshaped[self.elements]
        grad_u = torch.einsum('eki,ekqd->eqid', u_elements, self.basis_grads)
    
        F = self.I[None, None, :, :] + grad_u
        
        if self.METHOD == "Neo-Hookean":
            tr_grad_u = grad_u.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) # (e, q)
            det_grad_u = grad_u[:, :, 0, 0] * grad_u[:, :, 1, 1] - grad_u[:, :, 0, 1] * grad_u[:, :, 1, 0] # (e, q)
            det_F = 1.0 + tr_grad_u + det_grad_u # (e, q)
            det_F_expanded = det_F.unsqueeze(-1).unsqueeze(-1) # (e, q, 1, 1)
            J = det_F_expanded

            inv_F_T = torch.zeros_like(F)
            inv_F_T[:, :, 0, 0] = (1.0 + grad_u[:, :, 1, 1]) / det_F
            inv_F_T[:, :, 0, 1] = (-grad_u[:, :, 1, 0]) / det_F
            inv_F_T[:, :, 1, 0] = (-grad_u[:, :, 0, 1]) / det_F
            inv_F_T[:, :, 1, 1] = (1.0 + grad_u[:, :, 0, 0]) / det_F

            P = F * self.mu_quad + (0.5 * (det_F_expanded**2 - 1) * self.lmbda_quad - self.mu_quad) * inv_F_T
            
            sigma = (1.0 / J) * P @ F.transpose(-2, -1) # (e, q, 2, 2)

            sigma_avg_element = torch.mean(sigma, dim=1) # (e, 2, 2)
            s_xx, s_yy = sigma_avg_element[:, 0, 0], sigma_avg_element[:, 1, 1]
            s_xy = sigma_avg_element[:, 0, 1]
            
            von_mises_elemental = torch.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3*s_xy**2)

        elif self.METHOD == "St.Venant-Kirchhoff":
            J = torch.linalg.det(F).unsqueeze(-1).unsqueeze(-1)
            E = 0.5 * (F.transpose(-2,-1) @ F - self.I)
            tr_E = E.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

            S = self.lmbda_quad * tr_E * self.I + 2 * self.mu_quad * E
            sigma = (1/J) * F @ S @ F.transpose(-2,-1) # Cauchy stress

            sigma_avg_element = torch.mean(sigma, dim=1)
            s_xx, s_yy = sigma_avg_element[:, 0, 0], sigma_avg_element[:, 1, 1]
            s_xy = sigma_avg_element[:, 0, 1]
            
            von_mises_elemental = torch.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3*s_xy**2)

        else:
            raise ValueError(f"Unknown material model: {self.METHOD}")
        
        nodal_stress = torch.zeros(self.n_nodes, dtype=DTYPE, device=DEVICE)
        node_counts = torch.zeros(self.n_nodes, dtype=DTYPE, device=DEVICE)
        
        n_nodes_per_element = self.elements.shape[1]
        nodal_stress.scatter_add_(0, self.elements.flatten(), von_mises_elemental.repeat_interleave(n_nodes_per_element))
        node_counts.scatter_add_(0, self.elements.flatten(), torch.ones_like(self.elements, dtype=DTYPE).flatten())
        
        return nodal_stress / (node_counts + 1e-9)

    def compute_strain_energy(self, u_flat: torch.Tensor) -> torch.Tensor:
        """Computes the total strain energy of the system."""
        u = u_flat.reshape(self.n_nodes, 2)
        u_elements = u[self.elements]

        grad_u = torch.einsum('eki,ekqd->eqid', u_elements, self.basis_grads)
        F = self.I[None, None, :, :] + grad_u

        if self.METHOD == "Neo-Hookean":
            tr_grad_u = grad_u.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
            det_grad_u = grad_u[:, :, 0, 0] * grad_u[:, :, 1, 1] - grad_u[:, :, 0, 1] * grad_u[:, :, 1, 0]
            det_F = 1.0 + tr_grad_u + det_grad_u
            det_F_expanded = det_F.unsqueeze(-1).unsqueeze(-1)

            strain_energy_density = 0.5 * self.mu_quad * (torch.einsum('eqij,eqij->eq', F, F).unsqueeze(-1).unsqueeze(-1) - 3 - 2 * torch.log(det_F_expanded)) + 0.5 * self.lmbda_quad * (0.5 * (det_F_expanded**2 - 1) - torch.log(det_F_expanded))

        elif self.METHOD == "St.Venant-Kirchhoff":
            E = 0.5 * (F.transpose(-2, -1) @ F - self.I) # e, q, 2, 2
            tr_E = E.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1) # e, q, 1, 1

            strain_energy_density = (0.5 * self.lmbda_quad * tr_E**2 + self.mu_quad * torch.einsum('eqij,eqij->eqij', E, E))
            strain_energy_density = strain_energy_density.sum(dim=(-2, -1)) # e, q

        else:
            raise ValueError(f"Unknown material model: {self.METHOD}")


        # total_strain_energy = integral.assemble(self.basis, uh=self.basis.interpolate(strain_energy_density.squeeze().detach().cpu().numpy()))

        total_strain_energy = torch.einsum('eq,eq,q->', strain_energy_density.squeeze(), self.detJ, self.quad_weights)
        return total_strain_energy

    def compute_reaction_forces(self, u_flat: torch.Tensor) -> torch.Tensor:
        internal_forces = self.compute_internal_forces(u_flat.flatten())
        f_sum_top_x = torch.sum(internal_forces[self.top_nodes * 2])
        f_sum_top_y = torch.sum(internal_forces[self.top_nodes * 2 + 1])
        f_sum_btm_x = torch.sum(internal_forces[self.bottom_nodes * 2])
        f_sum_btm_y = torch.sum(internal_forces[self.bottom_nodes * 2 + 1])
        return torch.stack([f_sum_top_x, f_sum_btm_x, f_sum_top_y, f_sum_btm_y])

    def plot_materials(self):
        """Visualizes the assigned material distribution on the undeformed mesh."""
        print("--- Plotting material distribution ---")
        
        # Must use P1 vertices (self.mesh.p) and P1 topology (self.mesh.t) ---
        # tripcolor with shading='flat' cannot render 6-node P2 elements.
        
        # nodes_np = self.nodes.cpu().numpy() # This is P2 DOFs (wrong for tripcolor) if using P2
        # elements_np = self.elements.cpu().numpy() # This is P2 topology (n, 6) (wrong for tripcolor) if using P2
        
        nodes_np = self.mesh.p.T        # <-- P1 vertices 
        elements_np = self.mesh.t.T     # <-- P1 topology (n, 3) 
        
        # mu_elemental is per-element, so its (n_elements,) shape is perfect
        mu_elemental = torch.mean(self.mu_quad, dim=1).squeeze().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("Shear Modulus ($\mu$) Distribution")
        ax.set_aspect('equal')
        
        # This call will now work for both P1 and P2
        tpc = ax.tripcolor(nodes_np[:, 0], nodes_np[:, 1], elements_np, mu_elemental, shading='flat', cmap='GnBu')
        fig.colorbar(tpc, ax=ax, label="Shear Modulus ($\mu$)")
        plt.show()

    def _plot_material_comparison(self, save_dir, deformed_nodes, elements_p1_np, mu_elemental,
                                deformed_MNIST_X_true, deformed_MNIST_Y_true, 
                                elements_mnist_np, mu_map_np):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        
        # Predicted
        ax = axes[0]
        ax.set_title("$\mu$ on Deformed Shape (Predicted)")
        ax.set_aspect('equal')
        tpc00 = ax.tripcolor(deformed_nodes[:, 0], deformed_nodes[:, 1], elements_p1_np, mu_elemental, shading='flat', cmap='GnBu')
        fig.colorbar(tpc00, ax=ax, label="Shear Modulus ($\mu$)", shrink=0.8)

        # True
        ax = axes[1]
        ax.set_title("$\mu$ on Deformed Shape (True)")
        ax.set_aspect('equal')

        tpc10 = ax.tripcolor(deformed_MNIST_X_true, np.flip(deformed_MNIST_Y_true, [0]), 
                            elements_mnist_np, mu_map_np, shading='flat', cmap='GnBu')
        fig.colorbar(tpc10, ax=ax, label="Shear Modulus ($\mu$)", shrink=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fig1_material_comparison.png'))
        plt.close(fig)

    def _plot_ux_comparison(self, save_dir, deformed_MNIST_X_pred, deformed_MNIST_Y_pred, 
                            elements_mnist_np, MNIST_ux_pred, deformed_MNIST_X_true, 
                            deformed_MNIST_Y_true, dispX_true, error_X, RL2E_U_x):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)

        # X Disp ($U_x$) pred
        ax = axes[0]
        ax.set_title("X Disp ($U_x$) pred (interp.)")
        ax.set_aspect('equal')
        tpc02 = ax.tripcolor(deformed_MNIST_X_pred, deformed_MNIST_Y_pred, elements_mnist_np, MNIST_ux_pred, shading='gouraud', cmap='viridis')
        fig.colorbar(tpc02, ax=ax, shrink=0.8)

        # X Disp ($U_x$) true
        ax = axes[1]
        ax.set_title("X Disp ($U_x$) true")
        ax.set_aspect('equal')
        tpc03 = ax.tripcolor(deformed_MNIST_X_true, deformed_MNIST_Y_true, elements_mnist_np, dispX_true, shading='gouraud', cmap='viridis')
        fig.colorbar(tpc03, ax=ax, shrink=0.8)

        # X Disp Error
        ax = axes[2]
        ax.set_title("X Disp Error ($U_x^{true} - U_x^{pred}$)")
        ax.set_aspect('equal')
        tpc04 = ax.tripcolor(deformed_MNIST_X_true, deformed_MNIST_Y_true, elements_mnist_np, error_X, shading='gouraud', cmap='RdYlBu_r')
        ax.text(0.05, 0.05, f"RL2E = {RL2E_U_x:.2e}", transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.5))
        fig.colorbar(tpc04, ax=ax, shrink=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fig2_ux_comparison.png'))
        plt.close(fig)

    def _plot_uy_comparison(self, save_dir, deformed_MNIST_X_pred, deformed_MNIST_Y_pred, 
                            elements_mnist_np, MNIST_uy_pred, deformed_MNIST_X_true, 
                            deformed_MNIST_Y_true, dispY_true, error_Y, RL2E_U_y):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)

        # Y Disp ($U_y$) pred
        ax = axes[0]
        ax.set_title("Y Disp ($U_y$) pred (interp.)")
        ax.set_aspect('equal')
        tpc12 = ax.tripcolor(deformed_MNIST_X_pred, deformed_MNIST_Y_pred, elements_mnist_np, MNIST_uy_pred, shading='gouraud', cmap='viridis')
        fig.colorbar(tpc12, ax=ax, shrink=0.8)

        # Y Disp ($U_y$) true
        ax = axes[1]
        ax.set_title("Y Disp ($U_y$) true")
        ax.set_aspect('equal')
        tpc13 = ax.tripcolor(deformed_MNIST_X_true, deformed_MNIST_Y_true, elements_mnist_np, dispY_true, shading='gouraud', cmap='viridis')
        fig.colorbar(tpc13, ax=ax, shrink=0.8)

        # Y Disp Error
        ax = axes[2]
        ax.set_title("Y Disp Error ($U_y^{true} - U_y^{pred}$)")
        ax.set_aspect('equal')
        tpc14 = ax.tripcolor(deformed_MNIST_X_true, deformed_MNIST_Y_true, elements_mnist_np, error_Y, shading='gouraud', cmap='RdYlBu_r')
        ax.text(0.05, 0.05, f"RL2E = {RL2E_U_y:.2e}", transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.5))
        fig.colorbar(tpc14, ax=ax, shrink=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fig3_uy_comparison.png'))
        plt.close(fig)

    def _plot_strain_energy(self, save_dir, pred_load_steps, delta_strain_energy, 
                            true_load_steps, strain_energy_true):

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=100)
        RL2E_strain_energy = torch.linalg.norm(delta_strain_energy[-1] - torch.tensor(strain_energy_true[-1], dtype=DTYPE, device=DEVICE)) / torch.linalg.norm(torch.tensor(strain_energy_true[-1], dtype=DTYPE, device=DEVICE))
        ax.text(0.05, 0.95, f"RL2E = {RL2E_strain_energy:.2e}", transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title("Strain Energy ($\Delta \phi$) Comparison")
        ax.set_xlabel("Load Step")
        ax.set_ylabel("Strain Energy ($\Delta\phi$)")
        
        ax.plot(pred_load_steps, delta_strain_energy.detach().cpu().numpy(), 
                marker='o', linestyle='-', label='Predicted')
        
        ax.plot(true_load_steps, strain_energy_true, 
                marker='x', linestyle='None', label='True', color='red', markersize=10)
        
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fig4_strain_energy_comparison.png'))
        plt.close(fig)

    def _plot_reaction_forces(self, save_dir, pred_load_steps, pred_reaction_forces, 
                               true_load_steps, true_reaction_forces):
        """
        Plots the predicted vs true reaction forces for all 4 components.
        pred_reaction_forces: (n_steps, 4) tensor
        true_reaction_forces: (n_steps, 2) numpy array
        """
        pred_np = pred_reaction_forces.detach().cpu().numpy()
        true_np = true_reaction_forces
        titles = ["Top X Force", "Bottom X Force", "Top Y Force", "Bottom Y Force"]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=100)
        fig.suptitle("Reaction Force Comparison")
        for i in range(4):
            ax = axes[i]
            ax.set_title(titles[i])
            ax.set_xlabel("Load Step")
            ax.set_ylabel("Reaction Force")
            ax.plot(pred_load_steps, pred_np[:, i],
                    marker='o', linestyle='-', label='Predicted')
            if i == 0:
                # RL2E_top_x = np.linalg.norm(true_np[:, 0] - pred_np[:, 0], ord=2) / np.linalg.norm(true_np[-1, 0], ord=2)
                # ax.text(0.05, 0.95, f"RL2E = {RL2E_top_x:.2e}", transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.5))
                ax.plot(true_load_steps, true_np[:, 0],
                        marker='x', linestyle='None', label='True', color='red', markersize=10)
            elif i == 2:
                # RL2E_top_y = np.linalg.norm(true_np[:, 1] - pred_np[:, i], ord=2) / np.linalg.norm(true_np[-1, 1], ord=2)
                # ax.text(0.05, 0.95, f"RL2E = {RL2E_top_y:.2e}", transform=ax.transAxes, color='black', bbox=dict(facecolor='white', alpha=0.5))
                ax.plot(true_load_steps, true_np[:, 1],
                        marker='x', linestyle='None', label='True', color='red', markersize=10)
            ax.legend()
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(save_dir, 'fig5_reaction_force_comparison.png'))
        plt.close(fig)

    def plot_results(self, u_solved: torch.Tensor, stress_nodal: np.ndarray, 
                    material_map: torch.Tensor, dispX_true: np.array, 
                    dispY_true: np.array, strain_energy_true: np.array,
                    true_reaction_forces: np.array,
                    save_dir: str = 'results'): # Changed default save_dir to 'results'
        if u_solved is None:
            print("Solver failed, skipping visualization.")
            return
            
        os.makedirs(save_dir, exist_ok=True)
        print(f"--- Preparing data for plotting. Saving results to '{save_dir}' ---")

        u_solved_np = u_solved.cpu().numpy() # num_nodes, 2
        nodes_np = self.nodes.cpu().numpy() # num_nodes, 2
        elements_p1_np = self.mesh.t.T # (num_elements, 3)
        deformed_nodes = nodes_np + u_solved_np # num_nodes, 2
        
        MNIST_X = np.linspace(0, self.DOMAIN_SIZE-1, num=28)
        MNIST_Y = np.linspace(0, self.DOMAIN_SIZE-1, num=28)
        MNIST_GRID_X, MNIST_GRID_Y = np.meshgrid(MNIST_X+0.5, MNIST_Y+0.5, indexing='xy')
        MNIST_GRID = np.vstack([MNIST_GRID_X.flatten(), MNIST_GRID_Y.flatten()]).T # 784, 2
        
        triangulation_mnist = mtri.Triangulation(MNIST_GRID[:, 0], MNIST_GRID[:, 1])
        elements_mnist_np = triangulation_mnist.triangles

        # True deformation from dataset
        deformed_MNIST_X_true = MNIST_GRID[:, 0] + dispX_true # 784,
        deformed_MNIST_Y_true = MNIST_GRID[:, 1] + dispY_true # 784,

        # interpolated deformation from FEM to MNIST grid
        MNIST_ux_pred = self.basis.probes(MNIST_GRID.T) @ u_solved_np[:, 0] # 784,
        MNIST_uy_pred = self.basis.probes(MNIST_GRID.T) @ u_solved_np[:, 1] # 784,
        deformed_MNIST_X_pred = MNIST_GRID[:, 0] + MNIST_ux_pred
        deformed_MNIST_Y_pred = MNIST_GRID[:, 1] + MNIST_uy_pred

        # RL2E 
        error_X = dispX_true - MNIST_ux_pred
        error_Y = dispY_true - MNIST_uy_pred
        RL2E_U_x = np.linalg.norm(error_X, ord=2) / np.linalg.norm(dispX_true, ord=2)
        RL2E_U_y = np.linalg.norm(error_Y, ord=2) / np.linalg.norm(dispY_true, ord=2)
        
        # Elemental shear modulus (predicted)
        mu_elemental = torch.mean(self.mu_quad, dim=1).squeeze().cpu().numpy()

        # Lame parameters from true material map
        youngs_modulus_map = material_map.to(DTYPE).to(DEVICE) / 255.0 * (self.E_MODULUS_STIFF - self.E_MODULUS_SOFT) + self.E_MODULUS_SOFT
        mu_map = youngs_modulus_map / (2 * (1 + self.NU_POISSON))
        mu_map_np = mu_map.cpu().numpy()

        # Predicted strain energy
        delta_strain_energy = self.strain_energy_list - self.strain_energy_list[0]
        pred_load_steps = self.displacement_schedule.detach().cpu().numpy()

        # True strain energy load steps
        true_load_steps = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])

        # Predicted reaction forces
        pred_reaction_forces_tensor = torch.stack(self.reaction_force_list)

        self._plot_material_comparison(save_dir, deformed_nodes, elements_p1_np, mu_elemental,
                                    deformed_MNIST_X_true, deformed_MNIST_Y_true, 
                                    elements_mnist_np, mu_map_np)
        
        self._plot_ux_comparison(save_dir, deformed_MNIST_X_pred, deformed_MNIST_Y_pred, 
                                elements_mnist_np, MNIST_ux_pred, deformed_MNIST_X_true, 
                                deformed_MNIST_Y_true, dispX_true, error_X, RL2E_U_x)

        self._plot_uy_comparison(save_dir, deformed_MNIST_X_pred, deformed_MNIST_Y_pred, 
                                elements_mnist_np, MNIST_uy_pred, deformed_MNIST_X_true, 
                                deformed_MNIST_Y_true, dispY_true, error_Y, RL2E_U_y)
                                
        self._plot_strain_energy(save_dir, pred_load_steps, delta_strain_energy, 
                                true_load_steps, strain_energy_true)
        
        self._plot_reaction_forces(save_dir, pred_load_steps, pred_reaction_forces_tensor, 
                                   true_load_steps, true_reaction_forces)
        print(f"--- All plots saved to '{save_dir}' ---")
