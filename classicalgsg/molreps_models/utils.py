import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.numpy.linalg as la
import math
#local atomic environment
def fc(dist, cutoff):

    if dist > cutoff:
        return 0.0
    else:
        return 0.5 * np.cos(np.pi * dist/cutoff) + 0.5


def distance_matrix(positions):
    n_atoms = positions.shape[0]

    d = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms-1):
        d[i][i] = 0
        for j in range(i+1, n_atoms):
            d[i][j] = la.norm(positions[i, :]-positions[j, :])
            d[j][i] = d[i][j]
    return d

# def adjacency_matrix(positions, radial_cutoff):
#     n_atoms = positions.shape[0]

#     dd = np.zeros((n_atoms, n_atoms))

#     for i in range(n_atoms-1):
#         dd[i][i] = 0
#         for j in range(i+1, n_atoms):
#             dd[i][j] = fc(la.norm(positions[i, :] - positions[j, :]), radial_cutoff)
#             dd[j][i] = dd[i][j]
#     return np.array(dd)


def adjacency_matrix(positions, radial_cutoff):
    n_atoms = positions.shape[0]

    d = [[] for _ in range(n_atoms)]

    for i in range(n_atoms):
        for j in range(n_atoms):
            if i==j:
                d[i].append(0.0)
            else:
                d[i].append(fc(la.norm(positions[i, :] - positions[j, :]),
                               radial_cutoff))

    return np.array(d)



def angle(R_ij, R_ik):

    return np.dot(R_ij, R_ik) / (la.norm(R_ij) * la.norm(R_ik))


def angle_records(coords):

    num_atoms = coords.shape[0]
    atom_idxs = np.arange(num_atoms)

    angels = []
    for i in atom_idxs:
        neighbor_atom_idxs = np.delete(atom_idxs, i)
        atom_angles = {}
        for j in neighbor_atom_idxs:
            for k in neighbor_atom_idxs:
                if k > j:
                    theta = angle(coords[i]-coords[j], coords[i]-coords[k])
                    atom_angles.update({(j, k): theta})

        angels.append(atom_angles)

    return angels


def sco_to_boolean(scattering_operators):
    sco = [False, False, False]

    if scattering_operators.find('z') != -1:
        sco[0] = True

    if scattering_operators.find('f') != -1:
        sco[1] = True

    if scattering_operators.find('s') != -1:
        sco[2] = True

    return tuple(sco)