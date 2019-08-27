from firedrake import *


mesh = UnitIcosahedralSphereMesh(refinement_level=4)
x = SpatialCoordinate(mesh)
mesh.init_cell_orientations(x)


def get_p1_space():
    return FunctionSpace(mesh, "CG", 1)


def get_p1_prb_bcs():
    return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")


def p1_callback():
    P1 = get_p1_space()
    p = TrialFunction(P1)
    q = TestFunction(P1)
    return inner(grad(p), grad(q))*dx


def get_trace_nullspace(T):
    return VectorSpaceBasis(constant=True)


def get_coarse_nullspace():
    return VectorSpaceBasis(constant=True)


degree = 1
RT = FunctionSpace(mesh, "RT", degree)
DG = FunctionSpace(mesh, "DG", degree - 1)
W = RT * DG

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
n = FacetNormal(mesh)

f = Function(DG)
f.interpolate(x[0]*x[1]*x[2])

a = (dot(sigma, tau) - div(tau)*u + v*div(sigma)) * dx
L = f * v * dx

w = Function(W)
params = {'mat_type': 'matfree',
          # 'ksp_view': None,
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.HybridizationPC',
          'hybridization': {'ksp_type': 'cg',
                            'mat_type': 'matfree',
                            'ksp_rtol': 1e-8,
                            'ksp_monitor_true_residual': None,
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.GTMGPC',
                            'gt': {'mat_type': 'aij',
                                   'mg_levels': {'ksp_type': 'richardson',
                                                 'pc_type': 'bjacobi',
                                                 'sub_pc_type': 'ilu',
                                                 'ksp_max_it': 3},
                                   'mg_coarse': {'ksp_type': 'cg',
                                                 'ksp_monitor_true_residual': None,
                                                 'ksp_rtol': 1e-8,
                                                 'pc_type': 'gamg',
                                                 'mg_levels': {'ksp_type': 'chebyshev',
                                                               'pc_type': 'bjacobi',
                                                               'sub_pc_type': 'ilu',
                                                               'ksp_max_it': 4}}}}}
appctx = {'get_coarse_operator': p1_callback,
          'get_coarse_space': get_p1_space,
          'trace_nullspace': get_trace_nullspace,
          'get_coarse_op_nullspace': get_coarse_nullspace}

solve(a == L, w, solver_parameters=params, appctx=appctx)
sigma_h, u_h = w.split()

File("poisson_sphere.pvd").write(u_h)
