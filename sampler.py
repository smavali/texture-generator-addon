import numpy as np


class Random:
    def random_sampler(n, width, height, thresh, center):
        """
        params:
            n: number of samples
            width & height: dimentions of sample surface
            thresh: distance from edges
        return:
            list of points
        """
        x_center = center[0]
        y_center = center[1]

        xy_min = [x_center - width / 2 + thresh, y_center - height / 2 + thresh]
        xy_max = [x_center + width / 2 - thresh, y_center + height / 2 - thresh]
        data_np = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
        data = tuple(map(tuple, data_np))

        return list(data)


class PoissonDisc:
    def __init__(self, width=50, height=50, r=1, k=30):
        self.width, self.height = width, height
        self.r = r
        self.k = k
        self.samples_t = []

        # Cell side length
        self.a = r / np.sqrt(2)
        # Number of cells in the x- and y-directions of the grid
        self.nx, self.ny = int(width / self.a) + 1, int(height / self.a) + 1

        self.reset()

    def reset(self):
        """Reset the cells dictionary."""

        # A list of coordinates in the grid of cells
        coords_list = [(ix, iy) for ix in range(self.nx)
                       for iy in range(self.ny)]
        # Initilalize the dictionary of cells: each key is a cell's coordinates
        # the corresponding value is the index of that cell's point's
        # coordinates in the samples list (or None if the cell is empty).
        self.cells = {coords: None for coords in coords_list}

    def get_cell_coords(self, pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(pt[0] // self.a), int(pt[1] // self.a)

    def get_neighbours(self, coords):
        """Return the indexes of points in cells neighbouring cell at coords.
        For the cell at coords = (x,y), return the indexes of points in the
        cells with neighbouring coordinates illustrated below: ie those cells
        that could contain points closer than r.

                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo

        """

        dxdy = [(-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
                (-2, 0), (-1, 0), (1, 0), (2, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
                (-1, 2), (0, 2), (1, 2), (0, 0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < self.nx and
                    0 <= neighbour_coords[1] < self.ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = self.cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store the index of the contained point
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in
        its immediate neighbourhood.

        """

        cell_coords = self.get_cell_coords(pt)
        for idx in self.get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Squared distance between candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0] - pt[0]) ** 2 + (nearby_pt[1] - pt[1]) ** 2
            if distance2 < self.r ** 2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def get_point(self, refpt):
        """Try to find a candidate point near refpt to emit in the sample.

        We draw up to k points from the annulus of inner radius r, outer radius
        2r around the reference point, refpt. If none of them are suitable
        (because they're too close to existing points in the sample), return
        False. Otherwise, return the pt.

        """

        i = 0
        while i < self.k:
            rho, theta = (np.random.uniform(self.r, 2 * self.r),
                          np.random.uniform(0, 2 * np.pi))
            pt = refpt[0] + rho * np.cos(theta), refpt[1] + rho * np.sin(theta)
            if not (0 <= pt[0] < self.width and 0 <= pt[1] < self.height):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def sample(self, center, thresh):
        """Poisson disc random sampling in 2D.

        Draw random samples on the domain width x height such that no two
        samples are closer than r apart. The parameter k determines the
        maximum number of candidate points to be chosen around each reference
        point before removing it from the "active" list.

        """
        thresh = thresh
        width = self.width
        height = self.height

        current_center = (self.width / 2, self.height / 2)
        dx = center[0] - current_center[0]
        dy = center[1] - current_center[1]

        # Pick a random point to start with.
        pt = (np.random.uniform(0, self.width),
              np.random.uniform(0, self.height))
        self.samples = [pt]
        # Our first sample is indexed at 0 in the samples list...
        self.cells[self.get_cell_coords(pt)] = 0
        # and it is active, in the sense that we're going to look for more
        # points in its neighbourhood.
        active = [0]

        # As long as there are points in the active list, keep looking for
        # samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = np.random.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point relative to the reference point.
            pt = self.get_point(refpt)
            if pt:
                # Point pt is valid: add it to samples list and mark as active
                pt_t = (pt[0] + dx, pt[1] + dy)  # Transform point to the new center
                if pt[0] > thresh and pt[1] > thresh and pt[0] < width - thresh and pt[1] < height - thresh:
                    self.samples_t.append(pt_t)
                self.samples.append(pt)
                nsamples = len(self.samples) - 1
                active.append(nsamples)
                self.cells[self.get_cell_coords(pt)] = nsamples
            else:
                # We had to give up looking for valid points near refpt, so
                # remove it from the list of "active" points.
                active.remove(idx)

        return self.samples_t


class Aniso:

    def sample(self, width, height, spacing, center):

        points = []
        x_center = center[0]
        y_center = center[1]

        dx = x_center - width / 2
        dy = y_center - height / 2

        n_x = int(width / spacing)
        n_y = int(height / spacing)
        width_margin = (width - spacing * n_x) / 2
        height_margin = (height - spacing * n_y) / 2

        for i in range(n_x):
            for j in range(n_y):
                point = (dx + width_margin + i * spacing, dy + height_margin * j)
                points.append(point)

        return points


class R2:
    # Returns a pair of deterministic pseudo-random numbers
    # based on seed i=0,1,2,...
    def getU(self, i):
        useRadial = True  # user-set parameter

        # Returns the fractional part of (1+1/x)^y
        def fractionalPowerX(x, y):
            n = x * y
            a = np.zeros(n).astype(int)
            s = np.zeros(n).astype(int)
            a[0] = 1
            for j in range(y):
                c = np.zeros(n).astype(int)
                s[0] = a[0]
                for i in range(n - 1):
                    z = a[i + 1] + a[i] + c[i]
                    s[i + 1] = z % x
                    c[i + 1] = z / x  # integer division!
                a = np.copy(s)
            f = 0;
            for i in range(y):
                f += a[i] * pow(x, i - y)
            return f

        #
        u = np.zeros(2)
        v = np.zeros(2)
        v = [fractionalPowerX(2, i + 1), fractionalPowerX(3, i + 1)]
        if useRadial:
            u = [pow(v[0], 0.5) * np.cos(2 * np.pi * v[1]), pow(v[0], 0.5) * np.sin(2 * np.pi * v[1])]
        else:
            u = [v[0], v[1]]
        return u

    # Returns the i-th term of the canonical R2 sequence
    # for i = 0,1,2,...
    def r2(self, i):
        g = 1.324717957244746  # special math constant
        a = [1.0 / g, 1.0 / (g * g)]
        return [a[0] * (i + 1) % 1, a[1] * (1 + i) % 1]

    # Returns the i-th term of the jittered R2 (infinite) sequence.
    # for i = 0,1,2,...
    def jitteredR2(self, i):
        lambd = 1.0  # user-set parameter
        useRandomJitter = False  # user parameter option
        delta = 0.76  # empirically optimized parameter
        i0 = 0.300  # special math constant
        p = np.zeros(2)
        u = np.zeros(2)
        p = self.r2(i)
        if useRandomJitter:
            u = np.random.random(2)
        else:
            u = self.getU(i)
        k = lambd * delta * pow(np.pi, 0.5) / (
                4 * pow(i + i0, 0.5))  # only this line needs to be modified for point sequences
        j = [k * x for x in u]  # multiply array x by scalar k
        pts = [sum(x) for x in zip(p, j)]  # element-wise addition of arrays p and j
        return [s % 1 for s in pts]  # element-wise %1 for s
