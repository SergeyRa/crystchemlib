import core as ccl
import numpy as np
import pandas as pd
import streamlit as st
import pyxtal


@st.cache_data
def parsefile(file):
    parsed = {'datablock': [], 'structure': []}
    if file is not None:
        fromfile = ccl.parsecif(file)
        for n, d in zip(fromfile['name'], fromfile['data']):
            s = ccl.readstruct(d)
            if s is not None:
                parsed['datablock'].append(n)
                parsed['structure'].append(s)
    return parsed


st.write('The lower bound of structure determinants connecting given'
         ' structural units into the connected structure, referred to as'
         ' implicit hierarchical depth (IHD) of the structure, is dependent on'
         ' the rank of the symmetry group of the structure, the number of'
         ' orbits occupied by the structural units, and on their site-symmetry'
         ' groups (https://doi.org/10.1007/s11224-025-02638-9).')

col1, col2 = st.columns(2)
file = col1.file_uploader('Choose CIF file')
parsed = parsefile(file)
datablock = col2.selectbox('Choose datablock', parsed['datablock'])
st.divider()

structure = None
for d, s in zip(parsed['datablock'], parsed['structure']):
    if (d == datablock) and (s is not None):

        hall = pyxtal.symmetry.get_symmetry_from_ops(
            list(map(ccl.stringform, s.symops))
        )[0]
        G = pyxtal.symmetry.Group(hall, use_hall=True)
        col3, col4 = st.columns(2)
        col3.text(f'Space group {G.symbol} (#{G.number})'
                  '\n(derived from symmetry operations listed in CIF)')
        col3.write(f'Hall #{G.hall_number}')
        col3.write(f'General position {G[0].multiplicity}{G[0].letter}')

        lattice = np.array([
            ccl.orthonorm(s.cell, i) for i in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ])
        tol = 0.1  # tolerance in angstroms for Wyckoffs assignment
        WP = [G[0].merge(i.fract[:3], lattice, tol) for i in s.sites]

        wp_m = [i[1].multiplicity for i in WP]
        wp_l = [i[1].letter for i in WP]
        labels = [i.label for i in s.sites]
        X, Y, Z = np.round(np.array([i[0] for i in WP]).T, 6) % 1

        col4.dataframe(
            pd.DataFrame(
                {'label': labels, 'x': X, 'y': Y, 'z': Z,
                 'Wyckoff': [f'{i}{j}' for i, j in zip(wp_m, wp_l)]}
            ), hide_index=True
        )
        st.divider()

        ihd_data = pd.read_csv('ihd.csv', usecols=[1, 2, 3, 4, 6])
        ihd0 = ihd_data[
            (ihd_data['SG №'] == G.number)
            & ihd_data['WP Letter'].isin(wp_l)
        ]['IHD'].min()
        col3.write(f'IHD0 = {ihd0}')
        col3.write(f'IHD1 = {ihd0} + {len(WP)} - 1 = {ihd0 + len(WP) - 1}')

        gen_l = ihd_data[ihd_data['SG №'] == G.number]['WP Letter'].max()
        nongen = (np.array(wp_l) < gen_l).sum()
        col3.write(f'{nongen} special site(s):'
                   f' IHD {'=' if nongen <= 1 else '<='} IHD1')
        st.write('IHD reference table')
        st.dataframe(
            ihd_data[
                (ihd_data['SG №'] == G.number)
                & ihd_data['WP Letter'].isin(wp_l)
            ], hide_index=True
        )
