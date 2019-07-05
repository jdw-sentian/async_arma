import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

from conv import get_system_matrix

def poles(k):
    """Generates k pairs of poles in the unit circle
    Returns the resulting polynomial coefficients
    """

    pol = np.array([1])
    for _ in range(k):
        theta = np.random.random()*np.pi
        r = np.random.random()**0.5 # to get uniform over area
        coeffs = np.array([1, -2*np.cos(theta)*r, r**2])

        pol = np.convolve(pol, coeffs) # polynomial multiplication

    return pol

def get_ts(N, p=0, q=0):
    """
    p is the number of poles
    q is the number of nills
    """

    model = ArmaProcess(poles(p // 2), poles(q // 2))

    return model, model.generate_sample(N)

def resample(data, N=None, std_horizontal=0.1):
    """Resamples data in uniform random time points,
    using linear interpolation

    data: array-like
    N: integer, number of output samples

    Returns: array of new data, array of times
    """

    if N is None:
        N = len(data)

    #times = np.random.random([N])*N
    times = np.linspace(0, N, N)
    times = times + np.random.randn(N) * std_horizontal
    times = np.array(sorted(times))
    times = np.clip(times, 0, N-1e-6)
    return np.array([interpolate(data, t) for t in times]), times

def interpolate(data, x):
    """Return data[x], where x needn't be an integer
    Interpolated by linear interpolation
    """

    x0 = int(np.floor(x))
    x1 = min(int(np.ceil(x)), len(data)-1)

    return (x - x0)*data[x1] + (x1 - x)*data[x0]

def rmse(y, y_target):
    return np.sqrt(np.mean((y - y_target)**2))

def test_resampling():
    data = get_ts(100, p=2)
    irreg_data, times = resample(data)

    plt.plot(data, 'b')
    plt.plot(times, irreg_data, 'r')
    plt.scatter(times, irreg_data, color='r')
    plt.show()

def fit(data, times, p=2, fit_res=None):
    if fit_res is None:
        fit_res = p * 2
    width = p / fit_res
    A, b, times_red = get_system_matrix(times, data, fit_res, width)
    theta = np.linalg.lstsq(A, b, rcond=-1)
    theta = theta[0]
    print("Rec. using {0:d} piecewise, horizon {1:.3f}" \
          .format(fit_res, fit_res * width))
    return theta, A, b, times_red

def test_fit():
    np.random.seed(0)
    p = 4
    N = 1000
    model, orig_data = get_ts(N, p=p)
    print("AR({}), {} samples".format(p, N))
    times = np.linspace(0, N, N)

    fit_res = 10
    for std_horizontal in np.linspace(0, 1):
        data, times = resample(orig_data, std_horizontal=std_horizontal)

        theta, A, b, times_red = fit(data, times, p, fit_res)
        y_rec = np.dot(A, theta)

        err = rmse(b, y_rec)
        print("RMSE: {0:.3f}".format(err))
        burn_in = len(data) - len(y_rec)
        residuals = b - y_rec

        plt.cla()
        plt.hist(residuals, bins=50)
        plt.xlim([-10, 10])
        plt.title("Horz std: {0:.2f}, RMSE: {1:.3f}".format(std_horizontal, err))
        plt.draw()
        plt.pause(0.5)
    #plt.show()

    # signal to noise ratio



    plt.plot(times, data, 'b')
    plt.scatter(times, data, color='b')
    plt.plot(times_red, y_rec, 'r')
    plt.scatter(times_red, y_rec, color='r')

    plt.show()


if __name__ == '__main__':
    test_fit()