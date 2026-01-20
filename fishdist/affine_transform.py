# original code from C. Gohlke https://github.com/cgohlke/transformations/blob/deb1a195dab70f0f36365a104f9b70505e37b473/transformations/transformations.py#L920

# :License: BSD 3-Clause
# Copyright (c) 2006-2021, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import math
from scipy.ndimage import map_coordinates

from batoolset.img import Img

# _EPS = np.finfo(float).eps * 4.0
_EPS = np.finfo(float).eps
# _EPS = 0

# small change to the original method to make it easier to use for me
def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    return _affine_matrix_from_points(v0.T, v1.T, shear=shear, scale=scale, usesvd=usesvd)

# from https://github.com/cgohlke/transformations/blob/deb1a195dab70f0f36365a104f9b70505e37b473/transformations/transformations.py#L920 --> probably what I needed/wanted
def _affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError('input arrays are of wrong shape or type')

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims + 1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims + 1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims : 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right-handed system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M

# below is modified, not the completely C. Gohlke one
def superimposition_matrix(v0, v1, scale=True, usesvd=True):
    v0 = np.array(v0.T, dtype=np.float64, copy=False)[:3]
    v1 = np.array(v1.T, dtype=np.float64, copy=False)[:3]
    return affine_matrix_from_points(
        v0.T, v1.T, shear=False, scale=scale, usesvd=usesvd
    )

def quaternion_matrix(quaternion):

    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],
                q[1, 2] - q[3, 0],
                q[1, 3] + q[2, 0],
                0.0,
            ],
            [
                q[1, 2] + q[3, 0],
                1.0 - q[1, 1] - q[3, 3],
                q[2, 3] - q[1, 0],
                0.0,
            ],
            [
                q[1, 3] - q[2, 0],
                q[2, 3] + q[1, 0],
                1.0 - q[1, 1] - q[2, 2],
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def vector_norm(data, axis=None, out=None):
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    data *= data
    np.sum(data, axis=axis, out=out)
    np.sqrt(out, out)
    return None

'''
def translation_matrix(direction):
    """Return matrix to translate by direction vector.
    >>> v = np.random.random(3) - 0.5
    >>> np.allclose(v, translation_matrix(v)[:3, 3])
    True
    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M

def random_rotation_matrix(rand=None):
    """Return uniform random rotation matrix.
    rand: array like
        Three independent random variables that are uniformly distributed
        between 0 and 1 for each returned quaternion.
    >>> R = random_rotation_matrix()
    >>> np.allclose(np.dot(R.T, R), np.identity(4))
    True
    """
    return quaternion_matrix(random_quaternion(rand))


def random_quaternion(rand=None):
    """Return uniform random unit quaternion.
    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    >>> q = random_quaternion()
    >>> np.allclose(1, vector_norm(q))
    True
    >>> q = random_quaternion(np.random.random(3))
    >>> len(q.shape), q.shape[0]==4
    (1, True)
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        [
            np.cos(t2) * r2,
            np.sin(t1) * r1,
            np.cos(t1) * r1,
            np.sin(t2) * r2,
        ]
    )


def scale_matrix(factor, origin=None, direction=None):
    """Return matrix to scale by factor around origin in direction.
    Use factor -1 for point symmetry.
    >>> v = (np.random.rand(4, 5) - 0.5) * 20
    >>> v[3] = 1
    >>> S = scale_matrix(-1.234)
    >>> np.allclose(np.dot(S, v)[:3], -1.234*v[:3])
    True
    >>> factor = random.random() * 10 - 5
    >>> origin = np.random.random(3) - 0.5
    >>> direct = np.random.random(3) - 0.5
    >>> S = scale_matrix(factor, origin)
    >>> S = scale_matrix(factor, origin, direct)
    """
    if direction is None:
        # uniform scaling
        M = np.diag([factor, factor, factor, 1.0])
        if origin is not None:
            M[:3, 3] = origin[:3]
            M[:3, 3] *= 1.0 - factor
    else:
        # nonuniform scaling
        direction = unit_vector(direction[:3])
        factor = 1.0 - factor
        M = np.identity(4)
        M[:3, :3] -= factor * np.outer(direction, direction)
        if origin is not None:
            M[:3, 3] = (factor * np.dot(origin[:3], direction)) * direction
    return M


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    return None


def concatenate_matrices(*matrices):
    """Return concatenation of series of transformation matrices.
    >>> M = np.random.rand(16).reshape((4, 4)) - 0.5
    >>> np.allclose(M, concatenate_matrices(M))
    True
    >>> np.allclose(np.dot(M, M.T), concatenate_matrices(M, M.T))
    True
    """
    M = np.identity(4)
    for i in matrices:
        M = np.dot(M, i)
    return M


'''

# apply affine transform to a set of coords
def affineTransform(points, M):
    '''
    Returns transformed coordinates
    :param points: initial point cloud
    :param M: affine trafo matrix
    :return:
    '''
    # decompose the matrix into a transformation matrix and a translation
    transformation_matrix = M[:-1, :-1]
    # below is ok --> DO NOT CHANGE IT
    translation_vector = get_translation_vector_from_matrix(M) #M.T[-1][:-1]
    translation_vector = translation_vector[..., np.newaxis]
    transformed_coords = np.dot(transformation_matrix, points.T) + translation_vector
    return transformed_coords.T


# apply affine transform to an image
def affine_transform_3D_image(single_channel_3D_image, affine_transform_matrix, order=0):
    # see also https://nbviewer.org/gist/lhk/f05ee20b5a826e4c8b9bb3e528348688 for the rationale --> really good tuto
    # create coords for all the pixels of the current image
    coords = np.meshgrid(np.arange(single_channel_3D_image.shape[0]), np.arange(single_channel_3D_image.shape[1]), np.arange(single_channel_3D_image.shape[2]))
    zyx_coords = np.vstack([coords[0].reshape(-1), coords[1].reshape(-1), coords[2].reshape(-1)])
    # affine transform each coord of the image
    transformed = affineTransform(zyx_coords.T, affine_transform_matrix)
    # interpolate every new point
    new_img = map_coordinates(single_channel_3D_image, transformed.T,order=order)  # very useful function! good to know!
    # array is 1D --> reshape it to 3D
    new_img = new_img.reshape((single_channel_3D_image.shape[1], single_channel_3D_image.shape[0], single_channel_3D_image.shape[2]))
    # somehow the order of dims needs be corrected (some day understand why but ok for now!!!)
    new_img = np.moveaxis(new_img, 1, 0)
    return new_img


def get_translation_vector_from_matrix(M):
    return M.T[-1][:-1]



if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    v0 = np.array([[3531820.440, 1174966.736, 5162268.086],
                   [3531746.800, 1175275.159, 5162241.325],
                   [3532510.182, 1174373.785, 5161954.920],
                   [3532495.968, 1175507.195, 5161685.049],
                   [3532495.968, 1175507.195, 5161685.049]])
    v1 = np.array([[6089665.610, 3591595.470, 148.810],
                   [6089633.900, 3591912.090, 143.120],
                   [6089088.170, 3590826.470, 166.350],
                   [6088672.490, 3591914.630, 147.440],
                   [6088672.490, 3591914.630, 147.440]])

    # https://math.stackexchange.com/questions/15181/how-to-check-if-transformation-is-affine

    if False:
        v0 = np.asarray( [[0, 0], [1, 0], [0, 1], [1, 1]])
        v1 = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1.1]])

    if False:
        v0 = np.asarray( [[0, 0], [1, 0], [0, 1], [1, 1]]) # just a translation --> all is ok
        v1 = np.asarray( [[1, 1], [2, 1], [1, 2], [2, 2]])

    if False:
        v0 = np.asarray( [[0,0, 0], [0,1, 0], [0,0, 1], [0,1, 1]]) # just a translation --> all is ok
        # v1 = np.asarray( [[-2,1, 1], [-2,2, 1], [-2,1, 2], [-2,2, 2]])
        v1 = v0 + np.asarray([-2,1,1]).T # apply a -2,1,1 translation
        print('v1',v1)

    if True:
        v0 = np.asarray( [[0,0, 0], [0,1, 0], [0,0, 1], [0,1, 1]])
        M = Img('/E/Sample_images/FISH/Sample_transcription_dot_detection/manue_manually_segmented_images/training_set/tests/transmission_and_coloc_last_test_220916/acc.npy')
        v1 = affineTransform(v0,M)


    if False:
        # need remove ''' to execute this code !!!
        # test from C. Gohlke
        import random
        v0 = np.asarray([[0, 1031, 1031, 0], [0, 0, 1600, 1600]])
        v1 = np.asarray([[675, 826, 826, 677], [55, 52, 281, 277]])
        print(_affine_matrix_from_points(v0, v1))
        # array([[0.14549, 0.00062, 675.50008],
        #        [0.00048, 0.14094, 53.24971],
        #        [0., 0., 1.]])
        T = translation_matrix(np.random.random(3) - 0.5)
        R = random_rotation_matrix(np.random.random(3))
        S = scale_matrix(random.random())
        M = concatenate_matrices(T, R, S)
        v0 = (np.random.rand(4, 100) - 0.5) * 20
        v0[3] = 1
        v1 = np.dot(M, v0)
        v0[:3] += np.random.normal(0, 1e-8, 300).reshape(3, -1)
        M = _affine_matrix_from_points(v0[:3], v1[:3])
        print("ok2" if np.allclose(v1, np.dot(M, v0)) else "Error")

        # is that really working because the affine transform is not the same as what I initially had --> do I do mistakes
        print('M prior', M)
        # v0 = np.asarray([[0, 1031, 1031, 0], [0, 0, 1600, 1600]])
        # v1 = np.asarray([[675, 826, 826, 677], [55, 52, 281, 277]])
        v0 = v0.T
        v1=v1.T

    # if I prevent shear I get back the transform otherwise I don't get it --> why is that --> and all is svd independent --> is there a bug ???

    # la premiere colonne est pas retrouvée mais le reste si --> WHY ???
    M = affine_matrix_from_points(v0, v1, shear=True,scale=True, usesvd=True) # even if values aren't the same in the matrix that works... --> not sure I fully get it but ok for now
    # M = superimposition_matrix(v0, v1, scale=True, usesvd=False)
    print('affine trafo', M)

    print('get_translation_vector_from_matrix',get_translation_vector_from_matrix(M))

    # print('test',np.dot(M[:2,:2], v0.T)) # ça marche

    # I do have a bug now

    # M is correct but then there is a bug
    transformed_coords = affineTransform(v0, M)
    print('transformed_coords',transformed_coords)
    print("ok" if np.allclose(v1,transformed_coords) else "Error")  # passed --> not always passing but values are close



    # ok try read and understand that fully later maybe
    print(v1-transformed_coords) # --> that works even if the values of the transform aren't the same --> try understand why maybe some day

