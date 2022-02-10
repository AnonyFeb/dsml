import cvxpy as cp
import numpy as np
# import MOSEK as mosek
n = 20

x = 2 * (cp.Variable(n, boolean=True)- 0.5)
# eye = np.identity(n)
# xx = x.reshape((len(x), 1))
# xt = x.reshape((1, len(x)))
# xxt = np.dot(xx, xt)
v = 2 * (np.random.randint(0, 2, n) - 0.5)
vk =2 * ( np.random.randint(0, 2, n) - 0.5)
vj = v.reshape((len(v), 1))
vjt = v.reshape((1, len(v)))
vkt = vk.reshape(1, len(vk))
c = np.dot(vj, vjt) + np.dot(vj, vkt)

obj = cp.Minimize(cp.quad_form(x, c) + vk.T @ x)
# cons=[cp.matmul(np.ones(n),x) <=0, x@x.T==n]
# cons = [cp.matmul(np.ones(n), x) = 0, cp.matmul(np.ones(n), x) >= 0]
# cons = [eye @ xxt <= n, eye @ xxt >= n]
# A = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0.],
#               [0, 1, 0, 0, 0, 0, 1, 0, 0, 0.],
#               [0, 0, 1, 0, 0, 0, 0, 1, 0, 0.],
#               [0, 0, 0, 1, 0, 0, 0, 0, 1, 0.],
#               [1, 1, 1, 1, 1, 0, 0, 0, 0, 0.]])


# obj = cp.Minimize(disorders.T @ x)
# constraints = [cp.matmul(A, x) <= 1, cp.matmul(A, x) >= 1]
# prob = cp.Problem(obj, cons)
prob = cp.Problem(obj)
optimal_disorder = prob.solve(solver= cp.CPLEX, verbose=True)  # work
# optimal_disorder = prob.solve(solver= cp.XPRESS, verbose=True)   # work
# optimal_disorder = prob.solve(qcp=True, solver=cp.SCS, verbose=True)
# optimal_disorder = prob.solve(solver=cp.GLPK_MI, verbose=True)

print(f"Optimal disorder is {optimal_disorder}")
print(f"x: {x.value}")