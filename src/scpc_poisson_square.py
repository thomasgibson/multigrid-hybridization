from firedrake import *

mesh = UnitSquareMesh(10, 10)

def get_p1_space():
    return FunctionSpace(mesh, "CG", 1)

def get_p1_prb_bcs():
    return DirichletBC(get_p1_space(), Constant(0.0), "on_boundary")

def p1_callback():
    P1 = get_p1_space()
    p = TrialFunction(P1)
    q = TestFunction(P1)
    return inner(grad(p), grad(q))*dx

x = SpatialCoordinate(mesh)
degree = 1
n = FacetNormal(mesh)
U = FunctionSpace(mesh, "DRT", degree)
V = FunctionSpace(mesh, "DG", degree - 1)
T = FunctionSpace(mesh, "DGT", degree - 1)
W = U * V * T
u, p, lambdar = TrialFunctions(W)
w, q, gammar = TestFunctions(W)
f = Function(V)
f.interpolate(-2*(x[0]-1)*x[0] - 2*(x[1]-1)*x[1])
a = (dot(u, w)*dx - div(w)*p*dx
     + lambdar('+')*jump(w, n=n)*dS
     + div(u)*q*dx
     # Multiply transmission equation by -1 to ensure
     # SCPC produces the SPD operator after statically
     # condensing
     - gammar('+')*jump(u, n=n)*dS)
L = q*f*dx
s = Function(W)

params = {'ksp_type': 'preonly',
          'ksp_max_it': 10,
          # 'ksp_view': None,
          # 'ksp_monitor_true_residual': None,
          'mat_type': 'matfree',
          'pmat_type': 'matfree',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.SCPC',
          'pc_sc_eliminate_fields': '0, 1',
          'condensed_field': {'ksp_type': 'cg',
                              'mat_type': 'matfree',
                              'ksp_rtol': 1e-8,
                              'ksp_monitor_true_residual': None,
                              'pc_type': 'python',
                              'pc_python_type': 'firedrake.GTMGPC',
                              'gt': {'mg_levels': {'ksp_type': 'chebyshev',
                                                   'pc_type': 'jacobi',
                                                   'ksp_max_it': 4},
                                     'mg_coarse': {'ksp_type': 'preonly',
                                                   'ksp_rtol': 1e-8,
                                                   'pc_type': 'gamg',
                                                   'mg_levels': {'ksp_type': 'chebyshev',
                                                                 'pc_type': 'jacobi',
                                                                 'ksp_max_it': 4}}}}}
appctx = {'get_coarse_operator': p1_callback,
          'get_coarse_space': get_p1_space,
          'coarse_space_bcs': get_p1_prb_bcs()}
bcs = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")
problem = LinearVariationalProblem(a, L, s, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=params, appctx=appctx)
solver.solve()
