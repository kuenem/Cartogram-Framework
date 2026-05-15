import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geojson
import cvxpy as cp
from scipy.optimize import minimize, linprog
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean
import pulp


def polygon_area(pts):
    n = len(pts)
    A = 0
    for i in range(n):
        j = (i+1)%n
        A += pts[i,0]*pts[j,1] - pts[j,0]*pts[i,1]
    return 0.5*A


def polygon_areanp(points):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def signed_area(points):
    """Shoelace formula, returns signed area (positive for CCW)."""
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1] + x[-1]*y[0] - x[0]*y[-1])


def maxPolyAreaConvexOptim(points, target_area):
    if isinstance(points, list):
        points = np.array(points)
    # get number of points
    n = points.shape[0]
    
    x0 = points[:,0]
    y0 = points[:,1]

    # compute gradient of area
    gx = np.zeros(n)
    gy = np.zeros(n)

    for i in range(n):
        gx[i] = 0.5*(y0[(i+1)%n] - y0[(i-1)%n])
        gy[i] = 0.5*(x0[(i-1)%n] - x0[(i+1)%n])

    A0 = 0
    for i in range(n):
        j=(i+1)%n
        A0 += x0[i]*y0[j] - x0[j]*y0[i]
    A0 *= 0.5

    x = cp.Variable(n)
    y = cp.Variable(n)

    area_linear = A0 + gx @ (x-x0) + gy @ (y-y0)

    objective = cp.Minimize(
        cp.sum_squares(x-x0) + cp.sum_squares(y-y0)
    )

    constraints=[area_linear >= target_area]

    prob=cp.Problem(objective,constraints)
    prob.solve()

    return np.vstack((x.value,y.value)).T, area_linear.value


def maxPolyAreaConvexRelax(points, target_area):
    if isinstance(points, list):
        points = np.array(points)
    # get number of points
    n = points.shape[0]

    z0=points.flatten()
    m=2*n

    z=cp.Variable(m)
    X=cp.Variable((m,m),PSD=True)

    # area matrix
    Q=np.zeros((m,m))

    for i in range(n):
        j=(i+1)%n
        xi,yi=2*i,2*i+1
        xj,yj=2*j,2*j+1

        Q[xi,yj]+=0.5
        Q[xj,yi]-=0.5

    area=cp.trace(Q@X)

    # Schur complement
    constraints=[
        cp.bmat([
            [X,cp.reshape(z,(m,1))],
            [cp.reshape(z,(1,m)),np.ones((1,1))]
        ]) >> 0,
        area>=target_area
    ]

    objective=cp.Minimize(cp.sum_squares(z-z0))

    prob=cp.Problem(objective,constraints)
    prob.solve()

    return z.value.reshape((n,2)), area.value


def maxPolyAreaGeometricOptim(points, target_area):
    if isinstance(points, list):
        points = np.array(points)
    # get number of points
    n = points.shape[0]

    z0 = points.flatten()

    def polygon_area(z):
        pts = z.reshape(n,2)
        area = 0
        for i in range(n):
            j = (i+1)%n
            area += pts[i,0]*pts[j,1] - pts[j,0]*pts[i,1]
        return 0.5*area

    def objective(z):
        return np.sum((z - z0)**2)

    def constraint(z):
        return polygon_area(z) - target_area

    cons = {'type':'eq','fun':constraint}

    res = minimize(objective, z0, constraints=[cons])

    result = res.x.reshape(n,2)
    return result, polygon_area(result)


def maxPolyAreaConvexOptim2(points, target_area, max_iter=100):
    if isinstance(points, list):
        points = np.array(points)

    pts = points.copy()
    n = len(pts)

    for k in range(max_iter):

        x0 = pts[:,0]
        y0 = pts[:,1]

        # --- gradient ---
        gx = np.zeros(n)
        gy = np.zeros(n)
        for i in range(n):
            gx[i] = 0.5*(y0[(i+1)%n] - y0[(i-1)%n])
            gy[i] = 0.5*(x0[(i-1)%n] - x0[(i+1)%n])

        A0 = polygon_area(pts)

        # variables
        x = cp.Variable(n)
        y = cp.Variable(n)

        # linearized area
        area_linear = A0 + gx @ (x-x0) + gy @ (y-y0)

        # trust region (IMPORTANT)
        delta = 1.0
        constraints = [
            area_linear >= target_area,
            cp.abs(x - x0) <= delta,
            cp.abs(y - y0) <= delta
        ]

        obj = cp.Minimize(
            cp.sum_squares(x-x0) + cp.sum_squares(y-y0)
        )

        prob = cp.Problem(obj, constraints)
        prob.solve()

        if prob.status != "optimal":
            break

        pts = np.column_stack((x.value, y.value))

        # check convergence
        if abs(polygon_area(pts) - target_area) < 1e-3:
            break

    return pts, polygon_area(pts)


def maxPolyAreaConvexRelax2(points, target_area):
    if isinstance(points, list):
        points = np.array(points)

    n = len(points)
    z0 = points.flatten()
    m = 2*n

    z = cp.Variable(m)
    X = cp.Variable((m,m), PSD=True)

    Q = np.zeros((m,m))
    for i in range(n):
        j = (i+1)%n
        xi, yi = 2*i, 2*i+1
        xj, yj = 2*j, 2*j+1
        Q[xi,yj] += 0.5
        Q[xj,yi] -= 0.5

    area = cp.trace(Q @ X)

    constraints = []

    # Schur complement
    constraints.append(
        cp.bmat([
            [X, cp.reshape(z,(m,1))],
            [cp.reshape(z,(1,m)), np.ones((1,1))]
        ]) >> 0
    )

    # NEW: diagonal consistency (tightens relaxation)
    constraints.append(cp.diag(X) == cp.square(z))

    constraints.append(area >= target_area)

    obj = cp.Minimize(cp.sum_squares(z - z0))

    prob = cp.Problem(obj, constraints)
    prob.solve()

    return z.value.reshape(n,2), area.value


def PolyAreaQP(points, target_area):
    """
    Minimise sum of squared vertex displacements subject to area = target_area.
    Uses SLSQP (non‑convex constraint – local optimum).
    """
    points = np.array(points, dtype=float)
    n = len(points)
    x0 = points.flatten()  # [x1,y1,x2,y2,...]

    def obj(v):
        return np.sum((v - x0)**2)

    def area_constraint(v):
        # reshape to (n,2), compute signed area
        pts = v.reshape(n, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        a = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1] + x[-1]*y[0] - x[0]*y[-1])
        return a - target_area

    cons = {'type': 'eq', 'fun': area_constraint}
    res = minimize(obj, x0, method='SLSQP', constraints=cons, options={'ftol': 1e-8})
    new_pts = res.x.reshape(n, 2).tolist()
    return new_pts, signed_area(new_pts)


def PolyAreaQP2(points, target_area, objective='sum_squares', bounds=None):
    pts0 = np.array(points, dtype=float)
    n = len(pts0)
    x0 = pts0.flatten()

    if objective == 'sum_squares':
        def obj(v):
            return np.sum((v - x0)**2)
    else:
        raise ValueError(f"Objective {objective} not supported in QP version.")

    def area_con(v):
        pts = v.reshape(n,2)
        x, y = pts[:,0], pts[:,1]
        return 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1] + x[-1]*y[0] - x[0]*y[-1]) - target_area

    cons = [{'type': 'eq', 'fun': area_con}]
    bnds = None
    if bounds:
        dx_max, dy_max = bounds
        bnds = [(-dx_max, dx_max) for _ in range(2*n)]

    res = minimize(obj, x0, method='SLSQP', constraints=cons, bounds=bnds)
    new_pts = res.x.reshape(n,2).tolist()
    return new_pts, signed_area(new_pts)


def PolyAreaPenalty(points, target_area, lambda_=1000.0):
    """
    Minimise sum displacement^2 + lambda*(area - target)^2.
    Use BFGS (gradient estimated automatically).
    """
    points = np.array(points, dtype=float)
    n = len(points)
    x0 = points.flatten()

    def obj(v):
        pts = v.reshape(n, 2)
        x, y = pts[:, 0], pts[:, 1]
        A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1] + x[-1]*y[0] - x[0]*y[-1])
        return np.sum((v - x0)**2) + lambda_ * (A - target_area)**2

    res = minimize(obj, x0, method='BFGS', options={'gtol': 1e-8})
    new_pts = res.x.reshape(n, 2).tolist()
    return new_pts, signed_area(new_pts)


def PolyAreaPenalty2(points, target_area, objective='sum_squares', weight=1000.0, bounds=None):
    pts0 = np.array(points, dtype=float)
    n = len(pts0)
    x0 = pts0.flatten()

    if objective == 'sum_squares':
        def obj(v):
            pts = v.reshape(n,2)
            x,y = pts[:,0], pts[:,1]
            A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1] + x[-1]*y[0] - x[0]*y[-1])
            return np.sum((v - x0)**2) + weight * (A - target_area)**2
    else:
        raise ValueError("Objective not supported")

    bnds = None
    if bounds:
        dx_max, dy_max = bounds
        bnds = [(-dx_max, dx_max) for _ in range(2*n)]

    res = minimize(obj, x0, method='BFGS', bounds=bnds)
    new_pts = res.x.reshape(n,2).tolist()
    return new_pts, signed_area(new_pts)


def PolyAreaMinimalSurface(points, target_area, mu=0.1, lambda_=1000.0):
    """
    Minimise sum displacement^2 + mu*perimeter + lambda*(area-target)^2.
    Perimeter = sum of Euclidean edge lengths.
    """
    points = np.array(points, dtype=float)
    n = len(points)
    x0 = points.flatten()

    def obj(v):
        pts = v.reshape(n, 2)
        x, y = pts[:, 0], pts[:, 1]
        A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1] + x[-1]*y[0] - x[0]*y[-1])
        # perimeter (closed polygon)
        perim = np.sum(np.sqrt(np.diff(x, append=x[0])**2 + np.diff(y, append=y[0])**2))
        return np.sum((v - x0)**2) + mu * perim + lambda_ * (A - target_area)**2

    res = minimize(obj, x0, method='BFGS', options={'gtol': 1e-8})
    new_pts = res.x.reshape(n, 2).tolist()
    return new_pts, signed_area(new_pts)


def PolyAreaMinimalSurface2(points, target_area, objective='sum_squares', mu=0.1, weight=1000.0, bounds=None):
    pts0 = np.array(points, dtype=float)
    n = len(pts0)
    x0 = pts0.flatten()

    if objective == 'sum_squares':
        def obj(v):
            pts = v.reshape(n,2)
            x,y = pts[:,0], pts[:,1]
            A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1] + x[-1]*y[0] - x[0]*y[-1])
            perim = np.sum(np.sqrt(np.diff(x, append=x[0])**2 + np.diff(y, append=y[0])**2))
            return np.sum((v - x0)**2) + mu*perim + weight*(A - target_area)**2
    else:
        raise ValueError("Objective not supported")

    bnds = None
    if bounds:
        dx_max, dy_max = bounds
        bnds = [(-dx_max, dx_max) for _ in range(2*n)]

    res = minimize(obj, x0, method='BFGS', bounds=bnds)
    new_pts = res.x.reshape(n,2).tolist()
    return new_pts, signed_area(new_pts)


def PolyAreaLP2(points, target_area, objective='minimax', bounds=None):
    """
    Linear approximation of area via first‑order Taylor.
    Supports objectives:
        'minimax' – minimise max displacement (auxiliary variable)
        'sum_abs' – minimise sum |dx|+|dy|
    """
    pts0 = np.array(points, dtype=float)
    n = len(pts0)
    x0, y0 = pts0[:,0], pts0[:,1]

    # Compute partial derivatives of area at original points
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    for i in range(n):
        y_next = y0[(i+1)%n]
        y_prev = y0[(i-1)%n]
        grad_x[i] = 0.5 * (y_next - y_prev)
        x_next = x0[(i+1)%n]
        x_prev = x0[(i-1)%n]
        grad_y[i] = 0.5 * (x_prev - x_next)
    A0 = signed_area(pts0)

    # Linearised area: A0 + sum(grad_x*dx + grad_y*dy) = target
    # => sum(grad_x*dx + grad_y*dy) = target - A0

    # Create PuLP problem
    prob = pulp.LpProblem("PolyAreaLP", pulp.LpMinimize)

    # Decision variables: dx_i, dy_i (continuous, free)
    dx = [pulp.LpVariable(f"dx_{i}", lowBound=None, upBound=None) for i in range(n)]
    dy = [pulp.LpVariable(f"dy_{i}", lowBound=None, upBound=None) for i in range(n)]

    if objective == 'minimax':
        t = pulp.LpVariable("t", lowBound=0)
        prob += t
        # constraints: -t <= dx_i <= t, -t <= dy_i <= t
        for i in range(n):
            prob += dx[i] <= t
            prob += -dx[i] <= t
            prob += dy[i] <= t
            prob += -dy[i] <= t
    elif objective == 'sum_abs':
        # auxiliary variables for absolute values
        ax = [pulp.LpVariable(f"ax_{i}", lowBound=0) for i in range(n)]
        ay = [pulp.LpVariable(f"ay_{i}", lowBound=0) for i in range(n)]
        prob += pulp.lpSum(ax) + pulp.lpSum(ay)
        for i in range(n):
            prob += ax[i] >= dx[i]
            prob += ax[i] >= -dx[i]
            prob += ay[i] >= dy[i]
            prob += ay[i] >= -dy[i]
    else:
        raise ValueError("Objective must be 'minimax' or 'sum_abs'")

    # Area constraint (linearised)
    prob += pulp.lpSum(grad_x[i]*dx[i] + grad_y[i]*dy[i] for i in range(n)) == target_area - A0

    # Optional vertex movement bounds
    if bounds:
        dx_max, dy_max = bounds
        for i in range(n):
            prob += dx[i] >= -dx_max
            prob += dx[i] <= dx_max
            prob += dy[i] >= -dy_max
            prob += dy[i] <= dy_max

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob.status != 1:
        raise RuntimeError("LP infeasible or unbounded")

    dx_opt = [pulp.value(dx[i]) for i in range(n)]
    dy_opt = [pulp.value(dy[i]) for i in range(n)]
    new_pts = np.column_stack((x0 + dx_opt, y0 + dy_opt)).tolist()
    return new_pts, signed_area(new_pts)


def PolyAreaEdgeDirections(points, target_area):
    """
    Preserve edge directions, only change edge lengths.
    Area becomes a quadratic form in lengths.
    Solve QP with non‑negative lengths.
    """
    points = np.array(points, dtype=float)
    n = len(points)
    # Compute edge vectors and directions (unit vectors)
    edge_vecs = np.diff(points, axis=0, append=points[0:1])  # each row = (dx,dy)
    dirs = edge_vecs / np.linalg.norm(edge_vecs, axis=1, keepdims=True)
    # Variables: edge lengths l_i >= 0
    # Vertex coordinates: start at origin (or keep centroid fixed)
    # We'll keep centroid fixed to avoid translation.
    # Let new points = cumulative sum of l_i * dirs_i, then translate so centroid matches original.
    # Area expressed as function of l.
    # For a polygon with edge vectors v_i = l_i * d_i, signed area = 0.5 * sum over cross(v_i, v_{i+1})??? Actually:
    # area = 0.5 * |sum(x_i*y_{i+1} - x_{i+1}*y_i)|. Using cumulative sums, area becomes quadratic in l.
    # Simpler: solve directly the QP: minimise sum (l_i - l0_i)^2 subject to area(l)=target, l_i>=0.
    l0 = np.linalg.norm(edge_vecs, axis=1)

    def area_from_lengths(l):
        # build vertices from origin
        V = np.zeros((n+1, 2))
        for i in range(n):
            V[i+1] = V[i] + l[i] * dirs[i]
        # area of closed polygon V[0:n] (V[n]==V[0] if polygon closed? Actually last edge goes back to start)
        # But we built V from origin: V[n] may not equal V[0] unless closure constraint is enforced.
        # We need closure: sum(l_i * dirs_i) = 0. Enforce as equality constraint.
        # For area, use shoelace on V[0:n] (n points) assuming closed by last edge.
        x = V[:-1,0]; y = V[:-1,1]
        A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1] + x[-1]*y[0] - x[0]*y[-1])
        return A

    def closure(l):
        return np.sum(l[:, np.newaxis] * dirs, axis=0)  # should be (0,0)

    # Objective: min sum (l_i - l0_i)^2
    def obj(l):
        return np.sum((l - l0)**2)

    # Initial guess
    l0_guess = l0.copy()
    # Constraints
    cons = [{'type': 'eq', 'fun': lambda l: area_from_lengths(l) - target_area},
            {'type': 'eq', 'fun': lambda l: closure(l)[0], 'args': ()},
            {'type': 'eq', 'fun': lambda l: closure(l)[1], 'args': ()}]
    bounds = [(0, None) for _ in range(n)]
    res = minimize(obj, l0_guess, method='SLSQP', constraints=cons, bounds=bounds)
    if not res.success:
        raise RuntimeError("Optimisation failed: " + res.message)
    l_opt = res.x
    # Build new polygon vertices, then translate so centroid matches original centroid
    V = np.zeros((n+1, 2))
    for i in range(n):
        V[i+1] = V[i] + l_opt[i] * dirs[i]
    # Polygon consists of V[0:n] (n points) but V[0] is origin, last point may not equal origin.
    # We'll shift so that centroid of new polygon equals centroid of original.
    new_pts = V[:-1]
    centroid_new = np.mean(new_pts, axis=0)
    centroid_old = np.mean(points, axis=0)
    new_pts = new_pts + (centroid_old - centroid_new)
    return new_pts.tolist(), signed_area(new_pts.tolist())


def PolyAreaEdgeDirections2(points, target_area, objective='sum_squares', bounds=None):
    pts0 = np.array(points, dtype=float)
    n = len(pts0)
    edge_vecs = np.diff(pts0, axis=0, append=pts0[0:1])
    dirs = edge_vecs / np.linalg.norm(edge_vecs, axis=1, keepdims=True)
    l0 = np.linalg.norm(edge_vecs, axis=1)

    if objective == 'sum_squares':
        def obj(l):
            return np.sum((l - l0)**2)
    else:
        raise ValueError("Objective not supported")

    def closure_x(l):
        return np.sum(l * dirs[:,0])
    def closure_y(l):
        return np.sum(l * dirs[:,1])

    def area_con(l):
        V = np.zeros((n+1,2))
        for i in range(n):
            V[i+1] = V[i] + l[i] * dirs[i]
        V = V[:-1]
        x, y = V[:,0], V[:,1]
        A = 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1] + x[-1]*y[0] - x[0]*y[-1])
        return A - target_area

    cons = [{'type': 'eq', 'fun': closure_x},
            {'type': 'eq', 'fun': closure_y},
            {'type': 'eq', 'fun': area_con}]
    bounds_l = [(0, None) for _ in range(n)]
    res = minimize(obj, l0, method='SLSQP', constraints=cons, bounds=bounds_l)
    if not res.success:
        raise RuntimeError("Edge direction optimisation failed")
    l_opt = res.x
    V = np.zeros((n+1,2))
    for i in range(n):
        V[i+1] = V[i] + l_opt[i] * dirs[i]
    new_pts = V[:-1]
    centroid_orig = np.mean(pts0, axis=0)
    centroid_new = np.mean(new_pts, axis=0)
    new_pts = new_pts + (centroid_orig - centroid_new)
    return new_pts.tolist(), signed_area(new_pts.tolist())


def PolyAreaEdgeLP(points, target_area):
    
    n = len(points)
    dirs = []
    for i in range(n):
        d = np.array(points[(i+1)%n]) - np.array(points[i])
        dirs.append(d / np.linalg.norm(d))
    
    lengths = cp.Variable(n, nonneg=True)
    
    # reconstruct polygon
    x = [cp.Constant(points[0])]
    for i in range(n):
        x.append(x[-1] + lengths[i]*dirs[i])
    
    xs = cp.vstack(x[:-1])
    
    # linearized area approximation
    area = cp.sum(lengths)  # surrogate
    
    prob = cp.Problem(cp.Minimize((area - target_area)**2))
    prob.solve()
    
    pts = [xi.value for xi in xs]
    return pts, polygon_areanp(pts)


def PolyAreaMVEE(points, target_area):
    
    X = np.array(points)
    n = len(points)
    
    P = cp.Variable((2,2), PSD=True)
    c = cp.Variable(2)
    
    constraints = [
        cp.quad_form(X[i]-c, P) <= 1 for i in range(n)
    ]
    
    obj = cp.Minimize(-cp.log_det(P))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    scale = (target_area)**0.5
    new_points = c.value + scale*(X - c.value)
    
    return new_points, polygon_areanp(new_points)


def PolyAreaBBox(points, target_area):
    
    X = np.array(points)
    
    xmin = cp.Variable()
    xmax = cp.Variable()
    ymin = cp.Variable()
    ymax = cp.Variable()
    
    constraints = [
        xmin <= X[:,0], X[:,0] <= xmax,
        ymin <= X[:,1], X[:,1] <= ymax,
        (xmax - xmin)*(ymax - ymin) >= target_area
    ]
    
    prob = cp.Problem(cp.Minimize(xmax-xmin + ymax-ymin), constraints)
    prob.solve()
    
    scale = np.sqrt(target_area / polygon_areanp(points))
    return X*scale, polygon_areanp(X*scale)


def PolyAreaRadialLP(points, target_area):
    
    X = np.array(points)
    c = np.mean(X, axis=0)
    
    t = cp.Variable(len(points))
    
    new_pts = c + cp.multiply(t[:,None], (X - c))
    
    area = cp.sum(t)  # surrogate
    
    prob = cp.Problem(cp.Minimize((area-target_area)**2),
                      [t >= 0.5, t <= 2])
    prob.solve()
    
    return new_pts.value, polygon_areanp(new_pts.value)


def PolyAreaPerimeter(points, target_area):
    
    X = np.array(points)
    n = len(points)
    
    t = cp.Variable(n)
    
    perimeter = cp.sum(t)
    
    prob = cp.Problem(cp.Minimize((perimeter-target_area)**2),
                      [t >= 0])
    prob.solve()
    
    scale = np.mean(t.value)
    new_pts = X * scale
    return new_pts, polygon_areanp(new_pts)


def PolyAreaVertexBox(points, target_area):
    
    X = np.array(points)
    Y = cp.Variable(X.shape)
    
    constraints = [cp.abs(Y - X) <= 1.0]
    
    area = cp.sum(Y)  # surrogate
    
    prob = cp.Problem(cp.Minimize((area-target_area)**2), constraints)
    prob.solve()
    
    return Y.value, polygon_areanp(Y.value)


def PolyAreaRadialMVEE(points, target_area):
    
    X = np.array(points)
    c = np.mean(X, axis=0)
    
    t = cp.Variable(len(points))
    new_pts = c + cp.multiply(t[:,None], (X-c))
    
    P = cp.Variable((2,2), PSD=True)
    
    constraints = [
        cp.quad_form(new_pts[i]-c, P) <= 1
        for i in range(len(points))
    ]
    
    obj = cp.Minimize(-cp.log_det(P) + cp.sum_squares(t-1))
    prob = cp.Problem(obj, constraints + [t >= 0.5])
    prob.solve()
    
    return new_pts.value, polygon_areanp(new_pts.value)


def PolyAreaMVEE_fixed(points, target_area):
    
    X = np.array(points)
    
    P = cp.Variable((2,2), PSD=True)
    c = cp.Variable(2)
    
    constraints = []
    for i in range(len(points)):
        constraints.append(cp.quad_form(X[i] - c, P) <= 1)
    
    prob = cp.Problem(cp.Minimize(-cp.log_det(P)), constraints)
    prob.solve()
    
    # scale AFTER solving
    scale = np.sqrt(target_area / polygon_areanp(points))
    new_pts = c.value + scale*(X - c.value)


def PolyAreaRadialOptim(points, target_area):
    
    c = np.mean(points, axis=0)
    A0 = polygon_areanp(points)
    
    # convex 1D optimization (closed form)
    s = np.sqrt(target_area / A0)
    
    new_pts = [c + s*(np.array(p)-c) for p in points]
    return np.array(new_pts), polygon_areanp(new_pts)


def PolyAreaSequential(points, target_area, iters=10):
    
    X = np.array(points)
    c = np.mean(X, axis=0)
    
    t = np.ones(len(points))
    
    for _ in range(iters):
        new_pts = c + t[:,None]*(X-c)
        A = polygon_areanp(new_pts)
        
        scale = np.sqrt(target_area / A)
        t *= scale  # convex step
    
    new_pts = c + t[:,None]*(X-c)
    return new_pts, polygon_areanp(new_pts)

    
    return new_pts, polygon_areanp(new_pts)


def PolyAreaHullDeform(points, target_area):
    
    hull = ConvexHull(points)
    X = np.array(points)[hull.vertices]
    
    return PolyAreaSequential(X, target_area)


def PolyAreaLP(points, target_area):
    """
    Linear approximation of area around original points.
    Minimise sum |dx|+|dy| subject to linearised area = target.
    Uses scipy.linprog.
    """
    points = np.array(points, dtype=float)
    n = len(points)
    x0, y0 = points[:, 0], points[:, 1]

    # Compute partial derivatives of area wrt each coordinate
    # For a closed polygon: ∂A/∂xi = 0.5*(y_{i+1} - y_{i-1}), similarly for yi.
    # Indices modulo n.
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    for i in range(n):
        y_next = y0[(i+1) % n]
        y_prev = y0[(i-1) % n]
        grad_x[i] = 0.5 * (y_next - y_prev)
        x_next = x0[(i+1) % n]
        x_prev = x0[(i-1) % n]
        grad_y[i] = 0.5 * (x_prev - x_next)   # note sign

    A0 = signed_area(points)
    # Linearised area: A0 + sum(grad_x[i]*dx_i + grad_y[i]*dy_i) = target
    # Linear constraint: sum(grad_x*dx + grad_y*dy) = target - A0

    # Variables: dx_i, dy_i. We minimise sum |dx_i|+|dy_i| via auxiliary variables.
    # We have 2n variables (dx,dy) plus 2n aux variables (t_x, t_y). Total 4n.
    # Objective: sum(tx_i + ty_i)
    # Constraints: tx_i >= dx_i, tx_i >= -dx_i, similarly for ty.
    # plus linear area constraint: sum(grad_x*dx + grad_y*dy) = target - A0
    # No bounds on dx,dy (free), but can add small trust region if desired.

    c = np.zeros(4*n)
    c[:2*n] = 1.0   # coefficients for t variables (first 2n entries)

    # Inequality constraints: tx_i >= dx_i, tx_i >= -dx_i -> 2*2n = 4n constraints
    A_ub = []
    b_ub = []
    # for each i
    for i in range(n):
        # tx_i - dx_i >= 0
        row = np.zeros(4*n)
        row[i] = 1          # tx_i
        row[2*n + 2*i] = -1 # dx_i
        A_ub.append(row)
        b_ub.append(0)
        # tx_i + dx_i >= 0  (since tx_i >= -dx_i)
        row = np.zeros(4*n)
        row[i] = 1
        row[2*n + 2*i] = 1
        A_ub.append(row)
        b_ub.append(0)
        # similarly for ty_i
        j = n + i
        row = np.zeros(4*n)
        row[j] = 1          # ty_i
        row[2*n + 2*i + 1] = -1 # dy_i
        A_ub.append(row)
        b_ub.append(0)
        row = np.zeros(4*n)
        row[j] = 1
        row[2*n + 2*i + 1] = 1
        A_ub.append(row)
        b_ub.append(0)

    A_eq = []
    b_eq = []
    # area constraint
    row = np.zeros(4*n)
    for i in range(n):
        row[2*n + 2*i] = grad_x[i]       # dx_i
        row[2*n + 2*i + 1] = grad_y[i]   # dy_i
    A_eq.append(row)
    b_eq.append(target_area - A0)

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
    if not res.success:
        raise RuntimeError("LP failed: " + res.message)

    # Extract dx, dy
    dxy = res.x[2*n:]
    dx = dxy[0::2]
    dy = dxy[1::2]
    new_pts = np.column_stack((x0 + dx, y0 + dy)).tolist()
    return new_pts, signed_area(new_pts)


def PolyAreaUniformScale(points, target_area):
    """
    Scale entire polygon uniformly about its centroid.
    Shape is perfectly preserved; only size changes.
    """
    pts = np.array(points, dtype=float)
    A0 = signed_area(pts)
    if A0 == 0:
        raise ValueError("Original polygon has zero area – cannot scale.")
    scale = np.sqrt(abs(target_area / A0))
    # preserve orientation sign
    scale = scale if (target_area * A0 > 0) else -scale
    centroid = np.mean(pts, axis=0)
    new_pts = centroid + scale * (pts - centroid)
    return new_pts.tolist(), signed_area(new_pts.tolist())


def PolyAreaUniformScale2(points, target_area):
    pts0 = np.array(points, dtype=float)
    A0 = signed_area(pts0)
    if A0 == 0:
        raise ValueError("Zero area polygon cannot be scaled")
    scale = np.sqrt(abs(target_area / A0))
    if target_area * A0 < 0:
        scale = -scale
    centroid = np.mean(pts0, axis=0)
    new_pts = centroid + scale * (pts0 - centroid)
    return new_pts.tolist(), signed_area(new_pts.tolist())


def PolyAreaRadialScaling(points, target_area):
    
    c = np.mean(points, axis=0)
    A0 = polygon_areanp(points)
    s = np.sqrt(target_area / A0)
    
    new_points = [c + s*(np.array(p)-c) for p in points]
    return np.array(new_points), polygon_areanp(new_points)


def PolyAreaHullScale(points, target_area):
    
    hull = ConvexHull(points)
    hull_pts = np.array(points)[hull.vertices]
    
    return PolyAreaRadialScaling(hull_pts, target_area)


def PolyAreaCovariance(points, target_area):
    
    X = np.array(points)
    c = np.mean(X, axis=0)
    
    cov = np.cov(X.T)
    scale = np.sqrt(target_area / np.linalg.det(cov))
    
    new_pts = c + scale*(X-c)
    return new_pts, polygon_areanp(new_pts)