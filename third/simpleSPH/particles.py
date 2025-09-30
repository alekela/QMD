import numpy as np

from .data import field, Storage

def create(xmin, xmax, n, fields=[], nneibs=0):
    # xmin, xmax - bounds of the mesh
    # fields - list of the physical values
    # n - number of particles
    # nneibs - number of nearest neighbors to include
    # function returns Storage object with fields = [coords, size, element, neibs, ...]

    if not field.size in fields:
        fields.append(field.size)
    if not field.coords in fields:
        fields.append(field.coords)
    if not field.element in fields:
        fields.append(field.element)
    if not field.neibs in fields:
        fields.append(field.neibs)

    s = Storage(fields, n)
    particle_size    = (xmax - xmin) / n
    s[field.coords]  = np.linspace(xmin + 0.5 * particle_size, xmax - 0.5 * particle_size, n)
    s[field.size]    = particle_size*np.ones(n, dtype=field.fields_dtype[field.size])
    s[field.element] = np.zeros(n, dtype=field.fields_dtype[field.element])
    s[field.element][:] = "particle"

    if nneibs > 0:
        s[field.neibs] = np.zeros(n, dtype=(str(nneibs) + "i8"))
        for ineib in range(-nneibs//2, nneibs//2 + 1):
            if ineib < 0:
                s[field.neibs][:,ineib] = (np.arange(0, n) + ineib) % n
            if ineib > 0:
                s[field.neibs][:,ineib - 1] = (np.arange(0, n) + ineib) % n
    
    return s
