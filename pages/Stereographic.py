import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stereographic as sg
import streamlit as st


UB = np.array(
    st.text_area('Paste UB matrix/matrices here:').split()
).astype(float).reshape(-1, 3, 3)

c1, c2 = st.columns(2)
c1.text(f'{len(UB)} UB read')
if c1.toggle('Custom alignment') and (len(UB) != 0):
    al = c2.selectbox('Align UB (c along z, a* along x):',
                      np.arange(1, len(UB)+1).astype(int))
else:
    al = 0

if al != 0:
    R = sg.orient(UB[al-1])
else:
    R = np.eye(3)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, dpi=300)
for n, i in enumerate(UB):
    color = 'C' + str(n)
    A = R @ np.linalg.inv(i).T
    for j, lab in zip(A.T, 'abc'):
        t, r, h = sg.SGP(*j)
        ax.plot(t, r, h, color=color, fillstyle='none', alpha=0.75)
        ax.annotate(rf'${lab}_{n+1}$', (t, r), color=color,
                    xytext=(3, 3), textcoords='offset points')
ax.set_theta_zero_location('S')
ax.set_xticks(np.linspace(0, 330, 12) / 180 * np.pi)
ax.set_ylim(0, 1.1)
rad = np.array([30, 60, 90])  # radial ticks
ax.set_yticks(np.tan(rad/2 / 180 * np.pi),
              labels=rad.astype(str)+r'$\degree$')
ax.set_frame_on(False)
st.pyplot(fig)

if len(UB) > 0:
    At = [np.linalg.inv(i) for i in UB]
    At = np.vstack(At)
    labels = (np.array(['a', 'b', 'c'] * len(UB))
              + np.resize(np.arange(1, len(UB)+1),
                          (3, len(UB))).T.ravel().astype(str))
    angles = np.zeros((len(labels), len(labels)))
    for n, i in enumerate(At):
        for m, j in enumerate(At):
            angles[n, m] = np.acos(np.dot(i, j)
                                   / np.linalg.norm(i)
                                   / np.linalg.norm(j)) / np.pi * 180
    st.text('Angles between basis vectors (degrees):')
    st.dataframe(pd.DataFrame(np.round(angles, 1),
                              index=labels, columns=labels))
