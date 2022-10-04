import numpy as np
import nengo
import scipy

# Code for generating CS / CTX / encoders spaced evenly on the uniform hypersphere



"""
  Licensing:
    This code is distributed under the GNU LGPL license.

  Authors:
    Original FORTRAN77 version of i4_sobol by Bennett Fox.
    MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari

    Original Python version of is_prime by Corrado Chisari

    Original MATLAB versions of other functions by John Burkardt.
    PYTHON versions by Corrado Chisari

    Original code is available from http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
"""

def i4_bit_hi1(n):
    """
    i4_bit_hi1 returns the position of the high 1 bit base 2 in an integer.

    Example:
      +------+-------------+-----
      |    N |      Binary | BIT
      +------|-------------+-----
      |    0 |           0 |   0
      |    1 |           1 |   1
      |    2 |          10 |   2
      |    3 |          11 |   2
      |    4 |         100 |   3
      |    5 |         101 |   3
      |    6 |         110 |   3
      |    7 |         111 |   3
      |    8 |        1000 |   4
      |    9 |        1001 |   4
      |   10 |        1010 |   4
      |   11 |        1011 |   4
      |   12 |        1100 |   4
      |   13 |        1101 |   4
      |   14 |        1110 |   4
      |   15 |        1111 |   4
      |   16 |       10000 |   5
      |   17 |       10001 |   5
      | 1023 |  1111111111 |  10
      | 1024 | 10000000000 |  11
      | 1025 | 10000000001 |  11

    Parameters:
      Input, integer N, the integer to be measured.
      N should be nonnegative.  If N is nonpositive, the value will always be 0.

      Output, integer BIT, the number of bits base 2.
    """
    i = np.floor(n)
    bit = 0
    while (1):
        if (i <= 0):
            break
        bit += 1
        i = np.floor(i / 2.)
    return bit


def i4_bit_lo0(n):
    """
    I4_BIT_LO0 returns the position of the low 0 bit base 2 in an integer.

    Example:
      +------+------------+----
      |    N |     Binary | BIT
      +------+------------+----
      |    0 |          0 |   1
      |    1 |          1 |   2
      |    2 |         10 |   1
      |    3 |         11 |   3
      |    4 |        100 |   1
      |    5 |        101 |   2
      |    6 |        110 |   1
      |    7 |        111 |   4
      |    8 |       1000 |   1
      |    9 |       1001 |   2
      |   10 |       1010 |   1
      |   11 |       1011 |   3
      |   12 |       1100 |   1
      |   13 |       1101 |   2
      |   14 |       1110 |   1
      |   15 |       1111 |   5
      |   16 |      10000 |   1
      |   17 |      10001 |   2
      | 1023 | 1111111111 |   1
      | 1024 | 0000000000 |   1
      | 1025 | 0000000001 |   1

    Parameters:
      Input, integer N, the integer to be measured.
      N should be nonnegative.

      Output, integer BIT, the position of the low 1 bit.
    """
    bit = 0
    i = np.floor(n)
    while (1):
        bit = bit + 1
        i2 = np.floor(i / 2.)
        if (i == 2 * i2):
            break

        i = i2
    return bit


def i4_sobol_generate(dim_num, n, skip=1):
    """
    i4_sobol_generate generates a Sobol dataset.

    Parameters:
      Input, integer dim_num, the spatial dimension.
      Input, integer N, the number of points to generate.
      Input, integer SKIP, the number of initial points to skip.

      Output, real R(M,N), the points.
    """
    r = np.full((n, dim_num), np.nan)
    for j in range(n):
        seed = j + 1
        r[j, 0:dim_num], next_seed = i4_sobol(dim_num, seed)

    return r


def i4_sobol(dim_num, seed):
    """
    i4_sobol generates a new quasirandom Sobol vector with each call.

    Discussion:
      The routine adapts the ideas of Antonov and Saleev.

    Reference:
      Antonov, Saleev,
      USSR Computational Mathematics and Mathematical Physics,
      Volume 19, 1980, pages 252 - 256.

      Paul Bratley, Bennett Fox,
      Algorithm 659:
      Implementing Sobol's Quasirandom Sequence Generator,
      ACM Transactions on Mathematical Software,
      Volume 14, Number 1, pages 88-100, 1988.

      Bennett Fox,
      Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.

      Ilya Sobol,
      USSR Computational Mathematics and Mathematical Physics,
      Volume 16, pages 236-242, 1977.

      Ilya Sobol, Levitan,
      The Production of Points Uniformly Distributed in a Multidimensional
      Cube (in Russian),
      Preprint IPM Akad. Nauk SSSR,
      Number 40, Moscow 1976.

    Parameters:
      Input, integer DIM_NUM, the number of spatial dimensions.
      DIM_NUM must satisfy 1 <= DIM_NUM <= 40.

      Input/output, integer SEED, the "seed" for the sequence.
      This is essentially the index in the sequence of the quasirandom
      value to be generated.  On output, SEED has been set to the
      appropriate next value, usually simply SEED+1.
      If SEED is less than 0 on input, it is treated as though it were 0.
      An input value of 0 requests the first (0-th) element of the sequence.

      Output, real QUASI(DIM_NUM), the next quasirandom vector.
    """
    global atmost
    global dim_max
    global dim_num_save
    global initialized
    global lastq
    global log_max
    global maxcol
    global poly
    global recipd
    global seed_save
    global v

    if 'initialized' not in list(globals().keys()):
        initialized = 0
        dim_num_save = -1

    if (not initialized or dim_num != dim_num_save):
        initialized = 1
        dim_max = 40
        dim_num_save = -1
        log_max = 30
        seed_save = -1

#  Initialize (part of) V.
        v = np.zeros((dim_max, log_max))
        v[0:40, 0] = np.transpose([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        v[2:40, 1] = np.transpose([
                  1, 3, 1, 3, 1, 3, 3, 1,
            3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
            1, 3, 1, 3, 3, 1, 3, 1, 3, 1,
            3, 1, 1, 3, 1, 3, 1, 3, 1, 3])

        v[3:40, 2] = np.transpose([
                     7, 5, 1, 3, 3, 7, 5,
            5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
            5, 3, 3, 1, 7, 5, 1, 3, 3, 7,
            5, 1, 1, 5, 7, 7, 5, 1, 3, 3])

        v[5:40, 3] = np.transpose([
                                1, 7,  9,  13, 11,
            1, 3,  7,  9,  5,  13, 13, 11, 3,  15,
            5, 3,  15, 7,  9,  13, 9,  1,  11, 7,
            5, 15, 1,  15, 11, 5,  3,  1,  7,  9])

        v[7:40, 4] = np.transpose([
                                        9,  3,  27,
            15, 29, 21, 23, 19, 11, 25, 7,  13, 17,
            1,  25, 29, 3,  31, 11, 5,  23, 27, 19,
            21, 5,  1,  17, 13, 7,  15, 9,  31, 9])

        v[13:40, 5] = np.transpose([
                        37, 33, 7,  5,  11, 39, 63,
            27, 17, 15, 23, 29, 3,  21, 13, 31, 25,
            9,  49, 33, 19, 29, 11, 19, 27, 15, 25])

        v[19:40, 6] = np.transpose([
                                                   13,
            33, 115, 41, 79, 17, 29,  119, 75, 73, 105,
            7,  59,  65, 21, 3,  113, 61,  89, 45, 107])

        v[37:40, 7] = np.transpose([
            7, 23, 39])

#  Set POLY.
        poly = [
            1,   3,   7,   11,  13,  19,  25,  37,  59,  47,
            61,  55,  41,  67,  97,  91,  109, 103, 115, 131,
            193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299]

        atmost = 2 ** log_max - 1

#  Find the number of bits in ATMOST.
        maxcol = i4_bit_hi1(atmost)

#  Initialize row 1 of V.
        v[0, 0:maxcol] = 1


#  Things to do only if the dimension changed.
    if (dim_num != dim_num_save):

#  Check parameters.
        if (dim_num < 1 or dim_max < dim_num):
            print('I4_SOBOL - Fatal error!')
            print('  The spatial dimension DIM_NUM should satisfy:')
            print('    1 <= DIM_NUM <= %d' % dim_max)
            print('  But this input value is DIM_NUM = %d' % dim_num)
            return

        dim_num_save = dim_num

#  Initialize the remaining rows of V.
        for i in range(2, dim_num + 1):

#  The bits of the integer POLY(I) gives the form of polynomial I.
#  Find the degree of polynomial I from binary encoding.
            j = poly[i - 1]
            m = 0
            while (1):
                j = np.floor(j / 2.)
                if (j <= 0):
                    break
                m = m + 1

#  Expand this bit pattern to separate components of the logical array INCLUD.
            j = poly[i - 1]
            includ = np.zeros(m)
            for k in range(m, 0, -1):
                j2 = np.floor(j / 2.)
                includ[k - 1] = (j != 2 * j2)
                j = j2

#  Calculate the remaining elements of row I as explained
#  in Bratley and Fox, section 2.
            for j in range(m + 1, maxcol + 1):
                newv = v[i - 1, j - m - 1]
                l = 1
                for k in range(1, m + 1):
                    l = 2 * l
                    if (includ[k - 1]):
                        newv = np.bitwise_xor(
                            int(newv), int(l * v[i - 1, j - k - 1]))
                v[i - 1, j - 1] = newv

#  Multiply columns of V by appropriate power of 2.
        l = 1
        for j in range(maxcol - 1, 0, -1):
            l = 2 * l
            v[0:dim_num, j - 1] = v[0:dim_num, j - 1] * l

#  RECIPD is 1/(common denominator of the elements in V).
        recipd = 1.0 / (2 * l)
        lastq = np.zeros(dim_num)

    seed = int(np.floor(seed))

    if (seed < 0):
        seed = 0

    if (seed == 0):
        l = 1
        lastq = np.zeros(dim_num)

    elif (seed == seed_save + 1):

#  Find the position of the right-hand zero in SEED.
        l = i4_bit_lo0(seed)

    elif (seed <= seed_save):

        seed_save = 0
        l = 1
        lastq = np.zeros(dim_num)

        for seed_temp in range(int(seed_save), int(seed)):
            l = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(
                    int(lastq[i - 1]), int(v[i - 1, l - 1]))

        l = i4_bit_lo0(seed)

    elif (seed_save + 1 < seed):

        for seed_temp in range(int(seed_save + 1), int(seed)):
            l = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(
                    int(lastq[i - 1]), int(v[i - 1, l - 1]))

        l = i4_bit_lo0(seed)

#  Check that the user is not calling too many times!
    if (maxcol < l):
        print('I4_SOBOL - Fatal error!')
        print('  Too many calls!')
        print('  MAXCOL = %d\n' % maxcol)
        print('  L =      %d\n' % l)
        return

#  Calculate the new components of QUASI.
    quasi = np.zeros(dim_num)
    for i in range(1, dim_num + 1):
        quasi[i - 1] = lastq[i - 1] * recipd
        lastq[i - 1] = np.bitwise_xor(
            int(lastq[i - 1]), int(v[i - 1, l - 1]))

    seed_save = seed
    seed = seed + 1

    return [quasi, seed]


def i4_uniform(a, b, seed):
    """
    i4_uniform returns a scaled pseudorandom I4.

    Discussion:
      The pseudorandom number will be scaled to be uniformly distributed
      between A and B.

    Reference:
      Paul Bratley, Bennett Fox, Linus Schrage,
      A Guide to Simulation,
      Springer Verlag, pages 201-202, 1983.

      Pierre L'Ecuyer,
      Random Number Generation,
      in Handbook of Simulation,
      edited by Jerry Banks,
      Wiley Interscience, page 95, 1998.

      Bennett Fox,
      Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.

      Peter Lewis, Allen Goodman, James Miller
      A Pseudo-Random Number Generator for the System/360,
      IBM Systems Journal,
      Volume 8, pages 136-143, 1969.

    Parameters:
      Input, integer A, B, the minimum and maximum acceptable values.
      Input, integer SEED, a seed for the random number generator.

      Output, integer C, the randomly chosen integer.
      Output, integer SEED, the updated seed.
    """
    if (seed == 0):
        print('I4_UNIFORM - Fatal error!')
        print('  Input SEED = 0!')

    seed = np.floor(seed)
    a = round(a)
    b = round(b)

    seed = np.mod(seed, 2147483647)

    if (seed < 0):
        seed = seed + 2147483647

    k = np.floor(seed / 127773)

    seed = 16807 * (seed - k * 127773) - k * 2836

    if (seed < 0):
        seed = seed + 2147483647

    r = seed * 4.656612875E-10

#  Scale R to lie between A-0.5 and B+0.5.
    r = (1.0 - r) * (min(a, b) - 0.5) + r * (max(a, b) + 0.5)

#  Use rounding to convert R to an integer between A and B.
    value = round(r)

    value = max(value, min(a, b))
    value = min(value, max(a, b))

    c = value

    return [int(c), int(seed)]


def prime_ge(n):
    """
    PRIME_GE returns the smallest prime greater than or equal to N.

    Example:
      +-----+---------
      |   N | PRIME_GE
      +-----+---------
      | -10 |        2
      |   1 |        2
      |   2 |        2
      |   3 |        3
      |   4 |        5
      |   5 |        5
      |   6 |        7
      |   7 |        7
      |   8 |       11
      |   9 |       11
      |  10 |       11

    Parameters:
      Input, integer N, the number to be bounded.

      Output, integer P, the smallest prime number that is greater
      than or equal to N.
    """
    p = max(np.ceil(n), 2)
    while (not is_prime(p)):
        p = p + 1

    return p


def is_prime(n):
    """
    is_prime returns True if N is a prime number, False otherwise

    Parameters:
       Input, integer N, the number to be checked.

       Output, boolean value, True or False
    """
    if n != int(n) or n < 1:
        return False
    p = 2
    while p < n:
        if n % p == 0:
            return False
        p += 1
    return True



def spherical_transform(samples):
    """Map samples from the ``[0, 1]``--cube onto the hypersphere.

    Applies the `inverse transform method` to the distribution
    :class:`.SphericalCoords` to map uniform samples from the ``[0, 1]``--cube
    onto the surface of the hypersphere. [#]_

    Parameters
    ----------
    samples : ``(n, d) array_like``
        ``n`` uniform samples from the d-dimensional ``[0, 1]``--cube.

    Returns
    -------
    mapped_samples : ``(n, d+1) np.array``
        ``n`` uniform samples from the ``d``--dimensional sphere
        (Euclidean dimension of ``d+1``).

    See Also
    --------
    :class:`.Rd`
    :class:`.Sobol`
    :class:`.ScatteredHypersphere`
    :class:`.SphericalCoords`

    References
    ----------
    .. [#] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.

    Examples
    --------
    >>> from nengolib.stats import spherical_transform

    In the simplest case, we can map a one-dimensional uniform distribution
    onto a circle:

    >>> line = np.linspace(0, 1, 20)
    >>> mapped = spherical_transform(line)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(6, 3))
    >>> plt.subplot(121)
    >>> plt.title("Original")
    >>> plt.scatter(line, np.zeros_like(line), s=30)
    >>> plt.subplot(122)
    >>> plt.title("Mapped")
    >>> plt.scatter(*mapped.T, s=25)
    >>> plt.show()

    This technique also generalizes to less trivial situations, for instance
    mapping a square onto a sphere:

    >>> square = np.asarray([[x, y] for x in np.linspace(0, 1, 50)
    >>>                             for y in np.linspace(0, 1, 10)])
    >>> mapped = spherical_transform(square)

    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> plt.figure(figsize=(6, 3))
    >>> plt.subplot(121)
    >>> plt.title("Original")
    >>> plt.scatter(*square.T, s=15)
    >>> ax = plt.subplot(122, projection='3d')
    >>> ax.set_title("Mapped").set_y(1.)
    >>> ax.patch.set_facecolor('white')
    >>> ax.set_xlim3d(-1, 1)
    >>> ax.set_ylim3d(-1, 1)
    >>> ax.set_zlim3d(-1, 1)
    >>> ax.scatter(*mapped.T, s=15)
    >>> plt.show()
    """

    samples = np.asarray(samples)
    samples = samples[:, None] if samples.ndim == 1 else samples
    coords = np.empty_like(samples)
    n, d = coords.shape

    # inverse transform method (section 1.5.2)
    for j in range(d):
        coords[:, j] = SphericalCoords(d-j).ppf(samples[:, j])

    # spherical coordinate transform
    mapped = np.ones((n, d+1))
    i = np.ones(d)
    i[-1] = 2.0
    s = np.sin(i[None, :] * np.pi * coords)
    c = np.cos(i[None, :] * np.pi * coords)
    mapped[:, 1:] = np.cumprod(s, axis=1)
    mapped[:, :-1] *= c
    return mapped


class SphericalCoords(nengo.dists.Distribution):
    """Spherical coordinates for inverse transform method.

    This is used to map the hypercube onto the hypersphere and hyperball. [#]_

    Parameters
    ----------
    m : ``integer``
        Positive index for spherical coordinate.

    See Also
    --------
    :func:`.spherical_transform`
    :class:`nengo.dists.SqrtBeta`

    References
    ----------
    .. [#] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.

    Examples
    --------
    >>> from nengolib.stats import SphericalCoords
    >>> coords = SphericalCoords(3)

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 1, 1000)
    >>> plt.figure(figsize=(8, 8))
    >>> plt.subplot(411)
    >>> plt.title(str(coords))
    >>> plt.ylabel("Samples")
    >>> plt.hist(coords.sample(1000), bins=50, normed=True)
    >>> plt.subplot(412)
    >>> plt.ylabel("PDF")
    >>> plt.plot(x, coords.pdf(x))
    >>> plt.subplot(413)
    >>> plt.ylabel("CDF")
    >>> plt.plot(x, coords.cdf(x))
    >>> plt.subplot(414)
    >>> plt.ylabel("PPF")
    >>> plt.plot(x, coords.ppf(x))
    >>> plt.xlabel("x")
    >>> plt.show()
    """

    def __init__(self, m):
        super(SphericalCoords, self).__init__()
        self.m = m

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.m)

    def sample(self, n, d=None, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        shape = self._sample_shape(n, d)
        y = rng.uniform(size=shape)
        return self.ppf(y)

    def pdf(self, x):
        """Evaluates the PDF along the values ``x``."""
        return (np.pi * np.sin(np.pi * x) ** (self.m-1) /
                beta(self.m / 2., .5))

    def cdf(self, x):
        """Evaluates the CDF along the values ``x``."""
        y = .5 * betainc(self.m / 2., .5, np.sin(np.pi * x) ** 2)
        return np.where(x < .5, y, 1 - y)

    def ppf(self, y):
        """Evaluates the inverse CDF along the values ``x``."""
        y_reflect = np.where(y < .5, y, 1 - y)
        z_sq = scipy.special.betaincinv(self.m / 2., .5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < .5, x, 1 - x)


class Sobol(nengo.dists.Distribution):
    """Sobol sequence for quasi Monte Carlo sampling the ``[0, 1]``--cube.

    This is similar to ``np.random.uniform(0, 1, size=(num, d))``, but with
    the additional property that each ``d``--dimensional point is `uniformly
    scattered`.

    This is a wrapper around a library by the authors Corrado Chisari and
    John Burkardt (see `License <license.html>`__). [#]_

    See Also
    --------
    :class:`.Rd`
    :class:`.ScatteredCube`
    :func:`.spherical_transform`
    :class:`.ScatteredHypersphere`

    Notes
    -----
    This is **deterministic** for dimensions up to ``40``, although
    it should in theory work up to ``1111``. For higher dimensions, this
    approach will fall back to ``rng.uniform(size=(n, d))``.

    References
    ----------
    .. [#] http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html

    Examples
    --------
    >>> from nengolib.stats import Sobol
    >>> sobol = Sobol().sample(10000, 2)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(6, 6))
    >>> plt.scatter(*sobol.T, c=np.arange(len(sobol)), cmap='Blues', s=7)
    >>> plt.show()
    """

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            # Tile the points optimally. TODO: refactor
            return np.linspace(1./n, 1, n)[:, None]
        if d is None or not isinstance(d, int) or d < 1:
            # TODO: this should be raised when the ensemble is created
            raise ValueError("d (%d) must be positive integer" % d)
        if d > 40:
            warnings.warn("i4_sobol_generate does not support d > 40; "
                          "falling back to Monte Carlo method", UserWarning)
            return rng.uniform(size=(n, d))
        return i4_sobol_generate(d, n, skip=0)


def _rd_generate(n, d, seed=0.5):
    """Generates the first ``n`` points in the ``R_d`` sequence."""

    # http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    def gamma(d, n_iter=20):
        """Newton-Raphson-Method to calculate g = phi_d."""
        x = 1.0
        for _ in range(n_iter):
            x -= (x**(d + 1) - x - 1) / ((d + 1) * x**d - 1)
        return x

    g = gamma(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = (1/g) ** (j + 1) % 1

    z = np.zeros((n, d))
    z[0] = (seed + alpha) % 1
    for i in range(1, n):
        z[i] = (z[i-1] + alpha) % 1

    return z


class Rd(nengo.dists.Distribution):
    """Rd sequence for quasi Monte Carlo sampling the ``[0, 1]``--cube.

    This is similar to ``np.random.uniform(0, 1, size=(num, d))``, but with
    the additional property that each ``d``--dimensional point is `uniformly
    scattered`.

    This is based on the tutorial and code from [#]_. For `d=2` this is often
    called the Padovan sequence. [#]_

    See Also
    --------
    :class:`.Sobol`
    :class:`.ScatteredCube`
    :func:`.spherical_transform`
    :class:`.ScatteredHypersphere`

    References
    ----------
    .. [#] http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    .. [#] http://oeis.org/A000931

    Examples
    --------
    >>> from nengolib.stats import Rd
    >>> rd = Rd().sample(10000, 2)

    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(6, 6))
    >>> plt.scatter(*rd.T, c=np.arange(len(rd)), cmap='Blues', s=7)
    >>> plt.show()
    """  # noqa: E501

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            # Tile the points optimally. TODO: refactor
            return np.linspace(1./n, 1, n)[:, None]
        if d is None or not isinstance(d, int) or d < 1:
            # TODO: this should be raised when the ensemble is created
            raise ValueError("d (%d) must be positive integer" % d)
        return _rd_generate(n, d)


class ScatteredCube(nengo.dists.Distribution):
    """Number-theoretic distribution over the hypercube.

    Transforms quasi Monte Carlo samples from the unit hypercube
    to range between ``low`` and ``high``. These bounds may optionally be
    ``array_like`` with shape matching the sample dimensionality.

    Parameters
    ----------
    low : ``float`` or ``array_like``, optional
        Lower-bound(s) for each sample. Defaults to ``-1``.
    high : ``float`` or ``array_like``, optional
        Upper-bound(s) for each sample. Defaults to ``+1``.

    Other Parameters
    ----------------
    base : :class:`nengo.dists.Distribution`, optional
        The base distribution from which to draw `quasi Monte Carlo` samples.
        Defaults to :class:`.Rd` and should not be changed unless
        you have some alternative `number-theoretic sequence` over ``[0, 1]``.

    See Also
    --------
    :attr:`.cube`
    :class:`.Rd`
    :class:`.Sobol`
    :class:`.ScatteredHypersphere`

    Notes
    -----
    The :class:`.Rd` and :class:`.Sobol` distributions are deterministic.
    Nondeterminism comes from a random ``d``--dimensional shift (with
    wrap-around).

    Examples
    --------
    >>> from nengolib.stats import ScatteredCube
    >>> s1 = ScatteredCube([-1, -1, -1], [1, 1, 0]).sample(1000, 3)
    >>> s2 = ScatteredCube(0, 1).sample(1000, 3)
    >>> s3 = ScatteredCube([-1, .5, 0], [-.5, 1, .5]).sample(1000, 3)

    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> plt.figure(figsize=(6, 6))
    >>> ax = plt.subplot(111, projection='3d')
    >>> ax.scatter(*s1.T)
    >>> ax.scatter(*s2.T)
    >>> ax.scatter(*s3.T)
    >>> plt.show()
    """

    def __init__(self, low=-1, high=+1, base=Rd()):
        super(ScatteredCube, self).__init__()
        self.low = np.atleast_1d(low)
        self.high = np.atleast_1d(high)
        self.w = self.high - self.low
        self.base = base

    def __repr__(self):
        return "%s(low=%r, high=%r, base=%r)" % (
            type(self).__name__, self.low, self.high, self.base)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        u = self.base.sample(n, d, rng)

        # shift everything by the same random constant (with wrap-around)
        u = (u + rng.uniform(size=d)[None, :]) % 1.0

        return u * self.w[None, :] + self.low[None, :]


class ScatteredHypersphere(nengo.dists.UniformHypersphere):
    """Number--theoretic distribution over the hypersphere and hyperball.

    Applies the :func:`.spherical_transform` to the number-theoretic
    sequence :class:`.Rd` to obtain uniformly scattered samples.

    This distribution has the nice mathematical property that the
    `discrepancy` between the `empirical distribution` and :math:`n` samples
    is :math:`\\widetilde{\\mathcal{O}}\\left(\\frac{1}{n}\\right)` as opposed
    to :math:`\\mathcal{O}\\left(\\frac{1}{\\sqrt{n}}\\right)` for the `Monte
    Carlo` method. [#]_ This means that the number of samples are effectively
    squared, making this useful as a means for sampling ``eval_points`` and
    ``encoders`` in Nengo.

    See :doc:`notebooks/research/sampling_high_dimensional_vectors` for
    mathematical details.

    Parameters
    ----------
    surface : ``boolean``
        Set to ``True`` to restrict the points to the surface of the ball
        (i.e., the sphere, with one lower dimension). Set to ``False`` to
        sample from the ball. See also :attr:`.sphere` and :attr:`.ball` for
        pre-instantiated objects with these two options respectively.

    Other Parameters
    ----------------
    base : :class:`nengo.dists.Distribution`, optional
        The base distribution from which to draw `quasi Monte Carlo` samples.
        Defaults to :class:`.Rd` and should not be changed unless
        you have some alternative `number-theoretic sequence` over ``[0, 1]``.

    See Also
    --------
    :attr:`.sphere`
    :attr:`.ball`
    :class:`nengo.dists.UniformHypersphere`
    :class:`.Rd`
    :class:`.Sobol`
    :func:`.spherical_transform`
    :class:`.ScatteredCube`

    Notes
    -----
    The :class:`.Rd` and :class:`.Sobol` distributions are deterministic.
    Nondeterminism comes from a random ``d``--dimensional rotation
    (see :func:`.random_orthogonal`).

    The nengolib logo was created using this class with the Sobol sequence.

    References
    ----------
    .. [#] K.-T. Fang and Y. Wang, Number-Theoretic Methods in Statistics.
       Chapman & Hall, 1994.

    Examples
    --------
    >>> from nengolib.stats import ball, sphere
    >>> b = ball.sample(1000, 2)
    >>> s = sphere.sample(1000, 3)

    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> plt.figure(figsize=(6, 3))
    >>> plt.subplot(121)
    >>> plt.title("Ball")
    >>> plt.scatter(*b.T, s=10, alpha=.5)
    >>> ax = plt.subplot(122, projection='3d')
    >>> ax.set_title("Sphere").set_y(1.)
    >>> ax.patch.set_facecolor('white')
    >>> ax.set_xlim3d(-1, 1)
    >>> ax.set_ylim3d(-1, 1)
    >>> ax.set_zlim3d(-1, 1)
    >>> ax.scatter(*s.T, s=10, alpha=.5)
    >>> plt.show()
    """

    def __init__(self, surface, base=Rd()):
        super(ScatteredHypersphere, self).__init__(surface)
        self.base = base

    def __repr__(self):
        return "%s(surface=%r, base=%r)" % (
            type(self).__name__, self.surface, self.base)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            return super(ScatteredHypersphere, self).sample(n, d, rng)

        if self.surface:
            samples = self.base.sample(n, d-1, rng)
            radius = 1.
        else:
            samples = self.base.sample(n, d, rng)
            samples, radius = samples[:, :-1], samples[:, -1:] ** (1. / d)

        mapped = spherical_transform(samples)

        # radius adjustment for ball versus sphere, and a random rotation
        rotation = random_orthogonal(d, rng=rng)
        return np.dot(mapped * radius, rotation)
    
def random_orthogonal(d, rng=None):
    """Returns a random orthogonal matrix.
    Parameters
    ----------
    d : ``integer``
        Positive dimension of returned matrix.
    rng : :class:`numpy.random.RandomState` or ``None``, optional
        Random number generator state.
    Returns
    -------
    samples : ``(d, d) np.array``
        Random orthogonal matrix (an orthonormal basis);
        linearly transforms any vector into a uniformly sampled
        vector on the ``d``--ball with the same L2 norm.
    See Also
    --------
    :class:`.ScatteredHypersphere`
    Examples
    --------
    >>> from nengolib.stats import random_orthogonal, sphere
    >>> rng = np.random.RandomState(seed=0)
    >>> u = sphere.sample(1000, 3, rng=rng)
    >>> u[:, 0] = 0
    >>> v = u.dot(random_orthogonal(3, rng=rng))
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> ax = plt.subplot(111, projection='3d')
    >>> ax.scatter(*u.T, alpha=.5, label="u")
    >>> ax.scatter(*v.T, alpha=.5, label="v")
    >>> ax.patch.set_facecolor('white')
    >>> ax.set_xlim3d(-1, 1)
    >>> ax.set_ylim3d(-1, 1)
    >>> ax.set_zlim3d(-1, 1)
    >>> plt.legend()
    >>> plt.show()
    """

    rng = np.random if rng is None else rng
    m = nengo.dists.UniformHypersphere(surface=True).sample(d, d, rng=rng)
    u, s, v = scipy.linalg.svd(m)
    return np.dot(u, v)