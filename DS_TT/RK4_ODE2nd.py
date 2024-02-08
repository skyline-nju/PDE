def RK4_ODE2(x, v, h, F):
    """
    Use Runge-Kutta method to solve 2-nd ODE

    \dot{x} = v,
    \dot{v} = F(x, v)

    Here F dosen't depend on t explicitly.

    See details through the following link:
    https://www.mathworks.com/matlabcentral/answers/1702570-using-runge-kutta-algorithm-to-solve-second-order-ode
    However, for equation of dv3, it should be v+dv2/2, and for equation of dv4, it should be v+dv3
    """
    dx1 = h * v
    dv1 = h * F(x, v)

    x1 = x + 0.5 * dx1
    v1 = v + 0.5 * dv1

    dx2 = h * v1
    dv2 = h * F(x1, v1)

    x2 = x + 0.5 * dx2
    v2 = v + 0.5 * dv2

    dx3 = h * v2
    dv3 = h * F(x2, v2)

    x3 = x + dx3
    v3 = v + dv3

    dx4 = h * v3
    dv4 = h * F(x3, v3)

    dx = (dx1 + 2*dx2 + 2*dx3 + dx4)/6
    dv = (dv1 + 2*dv2 + 2*dv3 + dv4)/6
    return x + dx, v + dv


def RK4_ODE2nd_ge(x, v, h, i, F):
    """
    Use Runge-Kutta method to solve 2-nd ODE

    \dot{x} = v,
    \dot{v} = F(i, x, v)

    Here i is the index of t[:] with spacing dt = h / 2, i.e., t[i] = 2 * h * i

    See details through the following link:
    https://www.mathworks.com/matlabcentral/answers/1702570-using-runge-kutta-algorithm-to-solve-second-order-ode
    However, for equation of dv3, it should be v+dv2/2, and for equation of dv4, it should be v+dv3
    """
    dx1 = h * v
    dv1 = h * F(i, x, v)

    x1 = x + 0.5 * dx1
    v1 = v + 0.5 * dv1

    dx2 = h * v1
    dv2 = h * F(i+1, x1, v1)

    x2 = x + 0.5 * dx2
    v2 = v + 0.5 * dv2

    dx3 = h * v2
    dv3 = h * F(i+1, x2, v2)

    x3 = x + dx3
    v3 = v + dv3

    dx4 = h * v3
    dv4 = h * F(i+2, x3, v3)

    dx = (dx1 + 2*dx2 + 2*dx3 + dx4)/6
    dv = (dv1 + 2*dv2 + 2*dv3 + dv4)/6
    return x + dx, v + dv



def test_RK4():
    import matplotlib.pyplot as plt
    import numpy as np


    def fun(x, v):
        return -x

    h = 0.05
    t = np.arange(1000) * h
    
    x0 = 0
    v0 = 1

    x_arr = np.zeros_like(t)
    v_arr = np.zeros_like(t)

    x_arr[0] = x0
    v_arr[0] = v0

    for i in range(1, t.size):
        x_arr[i], v_arr[i] = RK4_ODE2(x_arr[i-1], v_arr[i-1], h, fun)
    
    plt.plot(t, x_arr)
    plt.plot(t, np.sin(t), "--")
    plt.show()
    plt.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np


    def fun(x, v):
        return -x

    h = 0.05
    t = np.arange(1000) * h
    
    x0 = 0
    v0 = 1

    x_arr = np.zeros_like(t)
    v_arr = np.zeros_like(t)

    x_arr[0] = x0
    v_arr[0] = v0

    for i in range(1, t.size):
        x_arr[i], v_arr[i] = RK4_ODE2(x_arr[i-1], v_arr[i-1], h, fun)
    
    plt.plot(t, x_arr)
    plt.plot(t, np.sin(t), "--")
    plt.show()
    plt.close()
