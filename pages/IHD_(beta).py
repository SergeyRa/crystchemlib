import core as ccl
import numpy as np
import pandas as pd
import streamlit as st
import spglib


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


col1, col2 = st.columns(2)
file = col1.file_uploader('Choose CIF file')
parsed = parsefile(file)
datablock = col2.selectbox('Choose datablock', parsed['datablock'])

structure = None
for d, s in zip(parsed['datablock'], parsed['structure']):
    if (d == datablock) and (s is not None):
        structure = s.p1()

        lattice = [ccl.orthonorm(structure.cell, i) for i in [[1, 0, 0, 1],
                                                              [0, 1, 0, 1],
                                                              [0, 0, 1, 1]]]
        positions = [i.fract[:3] for i in structure.sites]
        numbering = {j: i for i, j
                     in enumerate(set([k.symbol for k in structure.sites]))}
        numbers = [numbering[i.symbol] for i in structure.sites]
        symm = spglib.get_symmetry_dataset((lattice, positions, numbers),
                                           symprec=1e-3)
        sg = symm.number

        indep = sorted(set(symm.equivalent_atoms))
        wp_m = [(symm.equivalent_atoms == i).sum() for i in indep]
        wp_l = np.array([symm.wyckoffs[i] for i in indep])
        labels = [structure.sites[i].label for i in indep]
        symbols = [structure.sites[i].symbol for i in indep]
        X = [structure.sites[i].fract[0] for i in indep]
        Y = [structure.sites[i].fract[1] for i in indep]
        Z = [structure.sites[i].fract[2] for i in indep]

        summary = pd.DataFrame({
            'label': labels,
            'symbol': symbols,
            'x': X, 'y': Y, 'z': Z,
            'Wyckoff': [f'{i}{j}' for i, j in zip(wp_m, wp_l)]
        })

        st.write(f'Space group {symm.international} (#{sg}),'
                 f' {len(indep)} symmetry-independent site(s):')
        st.dataframe(summary, hide_index=True)

        ihd_data = pd.read_csv('ihd.csv')
        st.write('IHD reference table'
                 ' (https://doi.org/10.1007/s11224-025-02638-9):')
        st.dataframe(
            ihd_data[
                (ihd_data['SG №'] == sg) & ihd_data['WP Letter'].isin(wp_l)
            ],
            hide_index=True
        )
        ihd0 = ihd_data[
            (ihd_data['SG №'] == sg) & ihd_data['WP Letter'].isin(wp_l)
        ]['IHD'].min()
        st.write(f'IHD0 = {ihd0}')
        st.write(f'IHD1 = {ihd0} + {len(indep)} - 1 = {ihd0 + len(indep) - 1}')

        gen_l = ihd_data[ihd_data['SG №'] == sg]['WP Letter'].max()
        nongen = (wp_l < gen_l).sum()
        st.write(f'{nongen} non-general site(s):'
                 f' IHD {'=' if nongen <= 1 else '<='} IHD1')
