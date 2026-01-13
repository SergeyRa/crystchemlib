import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from tifffile import imread


st.text('Plot lattice(s) in reciprocal space reconstructions')
file = st.file_uploader("Choose unwarp image in TIFF format")

if file is not None:
    cm_name = st.sidebar.selectbox('Colormap', matplotlib.colormaps())
    cmap = matplotlib.colormaps[cm_name]
    scale = st.sidebar.selectbox('Scale',
                                 ('Linear', 'Logarithmic', 'Square root'))
    im = imread(file)
    im = im[::-1]
    im = im - im.min()  # set min to zero
    if scale == 'Logarithmic':
        im += np.sort(np.unique(im))[1]
        im = np.log(im)
    elif scale == 'Square root':
        im = im**0.5
    im = im / im.max()  # set scale to 1

    vmin, vmax = st.sidebar.slider('Intensity limits',
                                   0.0, 1.0, (0.0, 1.0), 0.01)
    cx = st.sidebar.number_input('Origin x, px',
                                 0, im.shape[0], im.shape[0]//2)
    cy = st.sidebar.number_input('Origin y, px',
                                 0, im.shape[1], im.shape[1]//2)
    c = np.array([cx, cy])
    xmin, xmax = st.sidebar.slider('x limits, px',
                                   0, im.shape[0], (0, im.shape[0]), 1)
    ymin, ymax = st.sidebar.slider('y limits, px',
                                   0, im.shape[1], (0, im.shape[1]), 1)
    dom = st.sidebar.selectbox('Domains', range(1, 4))
    N = st.sidebar.selectbox('Cells from origin', range(1, 5))
    sat = st.sidebar.toggle('Activate satellites')
    if sat:
        Nq = st.sidebar.selectbox('Max order of satellites', range(1, 5))

    domains = st.columns(dom)
    A = [0]*dom
    B = [0]*dom
    G = [0]*dom
    R = [0]*dom
    Qa = [0]*dom
    Qb = [0]*dom
    C = ['']*dom
    for n, d in enumerate(domains):
        A[n] = d.number_input(f'a*, px (domain {n+1})',
                              0.0, float(im.shape[0]/2),
                              float(im.shape[0]/10), 0.1)
        B[n] = d.number_input(f'b*, px (domain {n+1})',
                              0.0, float(im.shape[1]/2),
                              float(im.shape[0]/10), 0.1)
        G[n] = d.number_input(f'Gamma*, deg (domain {n+1})', 0, 179, 90)
        R[n] = d.number_input(f'Rotation, deg (domain {n+1})', -359, 359, 0)
        if sat:
            Qa[n] = d.number_input(f'q_a* (domain {n+1})', -1.0, 1.0, 0.0)
            Qb[n] = d.number_input(f'q_b* (domain {n+1})', -1.0, 1.0, 0.0)
        C[n] = d.selectbox(f'Color (domain {n+1})',
                           matplotlib.colors.CSS4_COLORS)

    fig, ax = plt.subplots(dpi=150)
    ax.imshow(im, cmap=cm_name, vmin=vmin, vmax=vmax)
    for n in range(len(domains)):
        a = np.array([A[n]*np.cos(R[n]*np.pi/180),
                      A[n]*np.sin(R[n]*np.pi/180)])
        b = np.array([B[n]*np.cos((G[n]+R[n])*np.pi/180),
                      B[n]*np.sin((G[n]+R[n])*np.pi/180)])
        for i in range(-N, N+1):
            ax.plot(*(np.array([-N*a, N*a]) + c + b*i).transpose(),
                    color=C[n], alpha=0.5)
            ax.plot(*(np.array([-N*b, N*b]) + c + a*i).transpose(),
                    color=C[n], alpha=0.5)
        if sat:
            q = np.array([[Qa[n]], [Qb[n]]])
            qpx = (np.array([a, b]) * q).sum(axis=0)
            sat_list = []
            for i in range(-N, N+1):
                for j in range(-N, N+1):
                    plt.plot(
                        *(
                            np.array([-Nq*qpx, Nq*qpx]) + c + a*i + b*j
                        ).transpose(), color=C[n], alpha=0.5, lw=1)
                    for k in range(-Nq, Nq+1):
                        if k != 0:
                            sat_list.append(c + a*i + b*j + qpx*k)
            ax.scatter(*np.array(sat_list).T, ec=C[n], fc='none', lw=0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    st.pyplot(fig)
