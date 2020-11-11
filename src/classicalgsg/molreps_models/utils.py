import numpy as np
import numpy.linalg as la


def fc(dist, cutoff):

    if dist > cutoff:
        return 0.0
    else:
        return 0.5 * np.cos(np.pi * dist/cutoff) + 0.5


def distance_matrix(positions):
    return la.norm(positions[:, None] - positions, dim=2, p=2)


def adjacency_matrix(positions, radial_cutoff):
    dist = distance_matrix(positions)
    dist = np.where(dist > radial_cutoff,
                    0.0,
                    0.5 * np.cos(np.pi * dist/radial_cutoff) + 0.5)
    return dist.fill_diagonal(0.0)


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


def scop_to_boolean(scattering_operators):
    sco = [False, False, False]

    if scattering_operators.find('z') != -1:
        sco[0] = True

    if scattering_operators.find('f') != -1:
        sco[1] = True

    if scattering_operators.find('s') != -1:
        sco[2] = True

    return tuple(sco)


def scop_to_str(scattering_operators):
    so_str = scattering_operators[1:-1].replace(',', '')
    return so_str.replace(' ', '')
