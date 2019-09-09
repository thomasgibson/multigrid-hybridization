"""
This module runs a convergence history for the mixed-hybrid methods
of a model elliptic problem (detailed in the main function).
"""

from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import numpy as np
import pandas as pd
import os


def run_mixed_hybrid_problem(r, degree, mixed_method, write=False):

    if mixed_method is None or mixed_method not in ("RT", "BDM"):
        raise ValueError("Must specify a method of 'RT' or 'BDM'")

    # Set up problem domain
    m = UnitSquareMesh(8, 8)
    mh = MeshHierarchy(m, r+1)

    # Set up function spaces
    if mixed_method == "RT":
        mesh = mh[-1]
        element = FiniteElement("RT", triangle, degree + 1)
        U = FunctionSpace(mesh, element)
        V = FunctionSpace(mesh, "DG", degree)
    else:
        assert mixed_method == "BDM"
        assert degree > 0, "Degree 0 is not valid for BDM method"
        mesh = mh[-1]
        element = FiniteElement("BDM", triangle, degree)
        U = FunctionSpace(mesh, element)
        V = FunctionSpace(mesh, "DG", degree - 1)

    def get_p1_space():
        return FunctionSpace(mesh, "CG", 1)

    def p1_callback():
        P1 = get_p1_space()
        p = TrialFunction(P1)
        q = TestFunction(P1)
        return inner(grad(p), grad(q))*dx

    def get_p1_prb_bcs():
        return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")

    x = SpatialCoordinate(mesh)
    W = U * V

    # Mixed space and test/trial functions
    W = U * V
    s = Function(W, name="solutions").assign(0.0)
    q, u = split(s)
    v, w = TestFunctions(W)

    a_scalar = sin(pi*x[0])*tan(pi*x[0]*0.25)*sin(pi*x[1])
    a_flux = -grad(a_scalar)

    f = Function(V).interpolate(-0.5*pi*pi*(4*cos(pi*x[0]) - 5*cos(pi*x[0]*0.5) + 2)*sin(pi*x[1]))

    a = (dot(q, v) - div(v)*u + div(q)*w)*dx

    L = w*f*dx
    F = a - L
    PETSc.Sys.Print("Solving hybrid-mixed system using static condensation.\n")
    params = {'snes_type': 'ksponly',
              'mat_type': 'matfree',
              'pmat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'ksp_rtol': 1e-8,
                                'ksp_monitor_true_residual': None,
                                'pc_type': 'python',
                                'pc_python_type': 'firedrake.GTMGPC',
                                'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                     'pc_type': 'jacobi',
                                                     'ksp_max_it': 2},
                                       'mg_coarse': {'ksp_type': 'preonly',
                                                     'pc_type': 'mg',
                                                     'mg_levels': {'ksp_type': 'chebyshev',
                                                                   'pc_type': 'jacobi',
                                                                   'ksp_max_it': 2}}}}}
    appctx = {'get_coarse_operator': p1_callback,
              'get_coarse_space': get_p1_space,
              'coarse_space_bcs': get_p1_prb_bcs()}
    problem = NonlinearVariationalProblem(F, s)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params,
                                        appctx=appctx)
    solver.solve()
    PETSc.Sys.Print("Solver finished.\n")

    outer_ksp = solver.snes.ksp
    ctx = outer_ksp.getPC().getPythonContext()
    inner_ksp = ctx.trace_ksp
    iterations = inner_ksp.getIterationNumber()

    # Computed flux, scalar, and trace
    q_h, u_h = s.split()

    # Now we compute the various metrics. First we
    # simply compute the L2 error between the analytic
    # solutions and the computed ones.
    scalar_error = errornorm(a_scalar, u_h, norm_type="L2")
    flux_error = errornorm(a_flux, q_h, norm_type="L2")

    # We keep track of all metrics using a Python dictionary
    error_dictionary = {"scalar_error": scalar_error,
                        "flux_error": flux_error}

    PETSc.Sys.Print("Finished test case for h=1/2^%d.\n" % r)

    # If write specified, then write output
    if write:
        File("results/hybrid_mixed/Hybrid-%s_deg%d.pvd" %
             (mixed_method, degree)).write(q_a, u_a, u_h)

    # Return all error metrics
    return error_dictionary, mesh, iterations


def run_mixed_hybrid_convergence(degree, method, start, end):

    PETSc.Sys.Print("Running convergence test for the hybrid-%s method "
                    "of degree %d"
                    % (method, degree))

    # Create arrays to write to CSV file
    r_array = []
    scalar_errors = []
    flux_errors = []
    flux_jumps = []
    num_cells = []
    iterations = []

    # Run over mesh parameters and collect error metrics
    for r in range(start, end + 1):
        r_array.append(r)
        error_dict, mesh, iters = run_mixed_hybrid_problem(r=r,
                                                           degree=degree,
                                                           mixed_method=method,
                                                           write=False)

        # Extract errors and metrics
        scalar_errors.append(error_dict["scalar_error"])
        flux_errors.append(error_dict["flux_error"])
        num_cells.append(mesh.num_cells())
        iterations.append(iters)

    PETSc.Sys.Print("Error in scalar: %0.13f" % scalar_errors[-1])
    PETSc.Sys.Print("Error in flux: %0.13f" % flux_errors[-1])

    if COMM_WORLD.rank == 0:
        degrees = [degree] * len(r_array)
        data = {"Mesh": r_array,
                "Degree": degrees,
                "NumCells": num_cells,
                "ScalarErrors": scalar_errors,
                "FluxErrors": flux_errors,
                "Iterations": iterations}
        df = pd.DataFrame(data)
        result = "H-%s-degree-%d.csv" % (method, degree)
        df.to_csv(result, index=False, mode="w")


# Test cases determined by (degree, mixed method)
test_cases = [(0, "RT"), (1, "RT"), (2, "RT"),
              (1, "BDM"), (2, "BDM")]


# Generates all CSV files by call the convergence test script
for test_case in test_cases:
    degree, method = test_case
    run_mixed_hybrid_convergence(degree, method,
                                 start=1, end=6)
