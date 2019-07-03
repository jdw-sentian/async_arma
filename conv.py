import numpy as np
import matplotlib.pyplot as plt

def get_system_matrix(x_data, y_data, p, width=1):
    """
    Design questions: 
    -what to do about burn-in?
        one option is to not use anything before y[0] + p*width
    -discard passages with too few data points?
    -optimization: throw out out-of-frame data

    Returns matrices A, b for linear system
    A*theta = b
    """
    A = []
    b = []
    times = []
    horizon = p*width

    x0 = x_data[0]
    x_d = []
    y_d = []
    for x_i, y_i in zip(x_data, y_data):
        if x_i < x0 + horizon:
            x_d.append(x_i)
            y_d.append(y_i)
            continue
        # remove data points that are too remote to have effect
        for i, x_di in enumerate(x_d):
            if x_di >= x_i - horizon:
                break
        x_d = x_d[i:]
        y_d = y_d[i:]
        if len(x_d) < 0:
            x_d.append(x_i)
            y_d.append(y_i)
            continue

        x = np.linspace(x_i - horizon, x_i, p)
        A.append(get_coeffs(x, x_d, y_d, width))
        b.append(y_i)
        times.append(x_i)

        x_d.append(x_i)
        y_d.append(y_i)

    return np.array(A), np.array(b), times

def get_coeffs(x, x_data, y_data, width=1):
    x_data = [min(x_data[0], x[0]) - width]*2 + list(x_data) + [max(x_data[-1], x[-1]) + width]*2
    y_data = [y_data[0]]*2 + list(y_data) + [y_data[-1]]*2
    y_chapeaus = []
    P = list(zip(*[x_data, y_data]))
    for idx, (p0, p1, p2) in enumerate(zip(P, P[1:], P[2:])):
        x0, _ = p0
        x1, y1 = p1
        x2, _ = p2
        #if idx == len(P) - 3:
        #    y_chapeaus.append(((x0, 0), p1, (x2, y1)))
        #else:
        y_chapeaus.append(((x0, 0), p1, (x2, 0)))

    coeffs = []
    for x_i in x:
        c = ((x_i - width/2, 0), (x_i, 1), (x_i + width/2, 0))
        coeffs.append(sum([chapeau_conv(c, c_j) for c_j in y_chapeaus]))

    return coeffs

def chapeau_conv(c0, c1):
    p00, p01, p02 = c0
    p10, p11, p12 = c1

    w00 = (p00, p01)
    w01 = (p01, p02)
    w10 = (p10, p11)
    w11 = (p11, p12)

    return trapezoid_conv(w00, w10) + \
           trapezoid_conv(w00, w11) + \
           trapezoid_conv(w01, w10) + \
           trapezoid_conv(w01, w11)

def trapezoid_conv(w0, w1):
    """Convolve two trapezoids,
    which need not have the same support
    """

    (x00, y00), (x01, y01) = w0
    (x10, y10), (x11, y11) = w1

    c = intersection((x00, x01), (x10, x11))
    if not len(c):
        return 0
    x0, x1 = c

    if x00 == x01 or x10 == x11:
        return 0

    z00 = ((x0 - x00)*y01 + (x01 - x0)*y00) / (x01 - x00)
    z01 = ((x1 - x00)*y01 + (x01 - x1)*y00) / (x01 - x00)
    z10 = ((x0 - x10)*y11 + (x11 - x0)*y10) / (x11 - x10)
    z11 = ((x1 - x10)*y11 + (x11 - x1)*y10) / (x11 - x10)

    return aligned_trapezoid_conv((z00, z01), (z10, z11)) * (x1 - x0)

def aligned_trapezoid_conv(y0, y1):
    y00, y01 = y0
    y10, y11 = y1

    # don't know if shoulkd be / 6 or /3
    return (y00*y10 + y01*y11 + (y00 + y01)*(y10 + y11)) / 3
    
def intersection(c0, c1):
    i0, j0 = c0
    i1, j1 = c1

    i, j = max(i0, i1), min(j0, j1)
    if i < j:
        return (i, j)
    else:
        return ()

def test_get_coeffs():
    scale = 10
    N = 5
    #x_data = np.random.random([N])*scale
    #x_data = np.array(sorted(x_data))
    x_data = np.linspace(0, scale, N)
    y_data = np.random.random([N])

    x = np.linspace(0, scale, scale)

    coeffs = get_coeffs(x, x_data, y_data, width=1)

    plt.plot(x_data, y_data, 'b')
    plt.scatter(x_data, y_data, color='b')
    plt.plot(x, coeffs, 'r')
    plt.scatter(x, coeffs, color='r')

    plt.draw()
    plt.pause(0.5)
    plt.cla()

if __name__ == '__main__':
    for _ in range(100):
        test_get_coeffs()