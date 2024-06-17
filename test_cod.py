import core
import matplotlib.pyplot as plt
import os

dir = './tests/COD_SO_notCH_34/'  # ends with slash
cifs = [i for i in os.listdir(dir) if i.endswith('.cif')]
N = len(cifs)
print(f'{N} files found')

structures = []
for i, c in enumerate(cifs):
    print(f'reading {i+1} file of {N}\r', end='')
    with open(dir+c) as f:
        dict = core.parsecif(f)
        for j in dict['data']:
            str = core.readstruct(j)
            if str is not None:
                structures.append(str)
M = len(structures)
print(f'\n{M} structures read')

polyhedra = []
for i, s in enumerate(structures):
    print(f'reading {i+1} structure of {M}\r', end='')
    centr = s.filter('symbol', ['S', 'S4+', 'S6+'])
    ligands = s.filter('symbol', ['O', 'O2-'])
    for j in centr:
        polyhedra.append(s.poly(j, ligands, 2))
P = len(polyhedra)
print(f'\n{P} polyhedra read')

distances = []
for i, p in enumerate(polyhedra):
    print(f'reading {i+1} polyhedron of {P}\r', end='')
    distances += p.listdist()['value']
D = len(distances)
print(f'\n{D} distances evaluated')
