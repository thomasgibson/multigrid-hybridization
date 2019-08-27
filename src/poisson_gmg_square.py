from firedrake import *


N = 3
m = UnitSquareMesh(N**2, N**2)
mh = MeshHierarchy(m, 5)
mesh = mh[-1]
x = SpatialCoordinate(mesh)


def get_p1_space():
    return FunctionSpace(mesh, "CG", 1)


def get_p1_prb_bcs():
    return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")


def p1_callback():
    P1 = get_p1_space()
    p = TrialFunction(P1)
    q = TestFunction(P1)
    return inner(grad(p), grad(q))*dx


degree = 1
RT = FunctionSpace(mesh, "RT", degree)
DG = FunctionSpace(mesh, "DG", degree - 1)
W = RT * DG

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)
n = FacetNormal(mesh)

f = Function(DG)
f.interpolate(2*pi**2 * sin(pi*x[0]) * sin(pi*x[1]))

a = (dot(sigma, tau) - div(tau)*u + v*div(sigma)) * dx
L = f * v * dx

w = Function(W)
params = {'mat_type': 'matfree',
          'ksp_view': None,
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.HybridizationPC',
          'hybridization': {'ksp_type': 'cg',
                            'mat_type': 'matfree',
                            'ksp_rtol': 1e-8,
                            'ksp_monitor_true_residual': None,
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.GTMGPC',
                            'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                 'pc_type': 'bjacobi',
                                                 'sub_pc_type': 'ilu',
                                                 'ksp_max_it': 2},
                                   'mg_coarse': {'ksp_type': 'cg',
                                                 'ksp_monitor_true_residual': None,
                                                 'ksp_rtol': 1e-8,
                                                 'pc_type': 'mg',
                                                 'pc_mg_type': 'full',
                                                 'mg_levels': {'ksp_type': 'richardson',
                                                               'pc_type': 'bjacobi',
                                                               'sub_pc_type': 'ilu',
                                                               'ksp_max_it': 3}}}}}
appctx = {'get_coarse_operator': p1_callback,
          'get_coarse_space': get_p1_space,
          'coarse_space_bcs': get_p1_prb_bcs()}

solve(a == L, w, solver_parameters=params, appctx=appctx)