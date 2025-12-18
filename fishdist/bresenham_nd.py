import numpy as np

def bresenham_nd(start, end):
    """
    Generate integer coordinates of a line between two points in N-dimensional space using the Bresenham algorithm.

    Args:
        start (array-like): Starting coordinate (int or float) of the line.
        end (array-like): Ending coordinate (int or float) of the line.

    Returns:
        np.ndarray: Array of integer coordinates representing the Bresenham line from start to end.

    Example:
        ```python
        >>> import numpy as np
        >>> bresenham_nd([0, 0, 0], [3, 2, 1])
        array([[0, 0, 0],
               [1, 1, 0],
               [2, 1, 1],
               [3, 2, 1]])

        ```
    """


    # Convert to integer grid
    p0 = np.rint(start).astype(int).copy()
    p1 = np.rint(end).astype(int).copy()

    # Number of dimensions
    ndim = len(p0)

    # Step direction for each dimension
    step = np.sign(p1 - p0)

    # Absolute differences
    dist = np.abs(p1 - p0)

    # Determine driving axis (longest delta)
    driving_axis = np.argmax(dist)

    # Initialize error terms
    # errors[i] tracks error for axis i w.r.t. driving_axis
    errors = 2 * dist.copy()
    main_delta = dist[driving_axis]

    # Result list
    points = [tuple(p0)]

    # Perform traversal
    while not np.array_equal(p0, p1):
        # Move along main axis
        p0[driving_axis] += step[driving_axis]

        # Update other axes
        for i in range(ndim):
            if i == driving_axis:
                continue
            if errors[i] >= main_delta:
                p0[i] += step[i]
                errors[i] -= 2 * main_delta
            errors[i] += 2 * dist[i]

        points.append(tuple(p0))

    return np.asarray(points)


if __name__ == '__main__':
    point1 =  (-1, 1, 1)
    point2 =  (5, 3, -1)

    point1 =  (0, 0, 0)
    point2 =  (3, 3.5, 0) # not working needs rint

    ListOfPoints = bresenham_nd(point1, point2)
    print(ListOfPoints)