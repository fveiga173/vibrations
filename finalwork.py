# Código corrigido e completo com perfis pré-definidos
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from PIL import Image

# Configuração inicial
d = st.sidebar
st.set_page_config(layout="wide")
st.title("Simulador de Vibração Veicular")
st.markdown("Modelo com carga na caçamba, chassi, motorista e eixos com molas.Feito por Felipe Veiga, como entre do Trabalho final da Disciplina de Vibrações Mecânicas - 15/07/2025.")

# Exibe imagem
d.header("Parâmetros do Sistema")
image = Image.open("modelo.jpg")
st.image(image, caption="Esquema do modelo físico", use_container_width=True)

# Perfis de veículo
definir = {
    "Carro": {"m1": 10, "m2": 40, "m3": 80, "m4": 30, "m5": 30,
               "k1": 1500, "k2": 1500, "k3": 800, "k4": 1200, "k5": 1200,
               "k6": 1500, "k7": 1500, "a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1, "e": 0.1, "f": 0.1, "g": 0.1},
    "Pickup": {"m1": 100, "m2": 300, "m3": 80, "m4": 60, "m5": 60,
                "k1": 3000, "k2": 3000, "k3": 1000, "k4": 2500, "k5": 2500,
                "k6": 3000, "k7": 3000, "a": 0.15, "b": 0.15, "c": 0.15, "d": 0.15, "e": 0.15, "f": 0.15, "g": 0.15},
    "Caminhão": {"m1": 500, "m2": 2000, "m3": 100, "m4": 150, "m5": 150,
                  "k1": 10000, "k2": 10000, "k3": 3000, "k4": 8000, "k5": 8000,
                  "k6": 10000, "k7": 10000, "a": 0.3, "b": 0.3, "c": 0.3, "d": 0.3, "e": 0.3, "f": 0.3, "g": 0.3}
}

if "valores" not in st.session_state:
    st.session_state.valores = definir["Carro"].copy()

perfil = d.selectbox("Escolha um perfil de veículo:", ["Personalizado"] + list(definir.keys()))
if perfil != "Personalizado" and d.button("Aplicar perfil selecionado"):
    st.session_state.valores = definir[perfil].copy()

# Entradas com valores vinculados ao estado
v = st.session_state.valores

labels = {
    "m1": "Massa da carga na caçamba (m1)",
    "m2": "Massa do chassi (m2)",
    "m3": "Massa do motorista (m3)",
    "m4": "Massa do eixo dianteiro (m4)",
    "m5": "Massa do eixo traseiro (m5)",
    "k1": "Rigidez da mola 1 (k1) - suporte carga esquerda",
    "k2": "Rigidez da mola 2 (k2) - suporte carga direita",
    "k3": "Rigidez da mola do assento (k3)",
    "k4": "Rigidez entre chassi e eixo dianteiro (k4)",
    "k5": "Rigidez dos pneus dianteiros (k5)",
    "k6": "Rigidez entre chassi e eixo traseiro (k6)",
    "k7": "Rigidez dos pneus traseiros (k7)",
    "a": "Distância a (entre chassi e mola k1)",
    "b": "Distância b (entre chassi e mola k2)",
    "c": "Distância c (entre carga e mola k6)",
    "d": "Distância d (entre carga e mola k4)",
    "e": "Distância e (entre carga e mola k1)",
    "f": "Distância f (entre carga e mola k2)",
    "g": "Distância g (entre carga e mola k3)"
}

for key in v:
    v[key] = d.number_input(labels.get(key, f"{key.upper()}:"), 0.01, 3000.0, value=float(v[key]), key=key)

m1, m2, m3, m4, m5 = v['m1'], v['m2'], v['m3'], v['m4'], v['m5']
k1, k2, k3, k4, k5, k6, k7 = v['k1'], v['k2'], v['k3'], v['k4'], v['k5'], v['k6'], v['k7']
a, b, c, d_, e, f_, g = v['a'], v['b'], v['c'], v['d'], v['e'], v['f'], v['g']

if d.button("Calcular e Simular"):
    symbs = sp.symbols('a b c d e f g k1 k2 k3 k4 k5 k6 k7 m1 m2 m3 m4 m5 I1 I2')
    vals = [a, b, c, d_, e, f_, g, k1, k2, k3, k4, k5, k6, k7, m1, m2, m3, m4, m5,
            (1/3)*m1*a**2, (1/3)*m2*b**2]
    subs = dict(zip(symbs, vals))

    xg2, tet2, xg1, tet1, x3, x4, x5 = sp.symbols('xg2 tet2 xg1 tet1 x3 x4 x5')
    ag2, ate2, ag1, ate1, a3, a4, a5 = sp.symbols('ag2 ate2 ag1 ate1 a3 a4 a5')

    xf = xg2 - a * tet2
    xh = xg2 + b * tet2
    xa = xg1 + c * tet1
    xb = xg1 + d_ * tet1
    xc = xg1 + e * tet1
    xd = xg1 + f_ * tet1
    xe = xg1 - g * tet1

    FM1 = k1 * (xf - xc)
    FM2 = k2 * (xh - xd)
    FM3 = k3 * xe
    FM4 = k4 * (xb - x4)
    FM5 = k5 * x4
    FM6 = k6 * (xa - x5)
    FM7 = k7 * x5

    eqs = [
        -FM6 + FM1 + FM2 + FM3 - FM4 - m2 * ag2,
        -FM6*c + FM1*a + FM2*b - FM3*g - (1/3)*m2*b**2 * ate2,
        -FM1 - FM2 - m1 * ag1,
        -FM1*a + FM2*b - (1/3)*m1*a**2 * ate1,
        -FM3 - m3 * a3,
        FM4 - FM5 - m4 * a4,
        FM6 - FM7 - m5 * a5
    ]

    VAR_D = [xg2, tet2, xg1, tet1, x3, x4, x5]
    VAR_A = [ag2, ate2, ag1, ate1, a3, a4, a5]

    EQ = [-sp.expand(eq) for eq in eqs]
    for i in range(7):
        for j in range(7):
            EQ[i] = sp.collect(EQ[i], VAR_D[j])

    Mmat = sp.zeros(7)
    Kmat = sp.zeros(7)
    for i in range(7):
        Mmat[i, i] = EQ[i].coeff(VAR_A[i], 1)
        for j in range(7):
            Kmat[i, j] = EQ[i].coeff(VAR_D[j], 1)

    Mnum = np.array(Mmat.subs(subs).evalf().tolist(), dtype=float)
    Knum = np.array(Kmat.subs(subs).evalf().tolist(), dtype=float)

    A = np.linalg.pinv(Mnum) @ Knum
    VAL, VET = np.linalg.eig(A)
    wn = np.sqrt(np.abs(VAL))
    fn = wn / (2 * np.pi)
    valid = wn > 1e-5
    wn, fn = wn[valid], fn[valid]
    VET = VET[:, valid]

    st.subheader("Frequências Naturais (Hz)")
    st.write(fn)

    t = np.linspace(0, 5, 5000)
    X0 = np.linspace(0.01, 0.05, VET.shape[0]).reshape(-1, 1)
    V0 = np.linspace(1, 5, VET.shape[0]).reshape(-1, 1)
    freq_inv = np.diag(np.where(wn != 0, 1 / wn, 0))
    Mdiag = VET.T @ Mnum @ VET
    VETn = VET / np.sqrt(np.diag(Mdiag))[np.newaxis, :]
    MAT1 = VETn.T @ Mnum @ X0
    MAT2 = freq_inv @ VETn.T @ Mnum @ V0
    Cs = np.cos(np.outer(wn, t))
    Sn = np.sin(np.outer(wn, t))
    response = VETn @ (Cs * MAT1 + Sn * MAT2)

    st.subheader("Resposta em Vibração Livre")
    nomes = [
        "Deslocamento do chassi (xg2)",
        "Rotação do chassi (θ2)",
        "Deslocamento da carga (xg1)",
        "Rotação da carga (θ1)",
        "Deslocamento do motorista (x3)",
        "Deslocamento do eixo dianteiro (x4)",
        "Deslocamento do eixo traseiro (x5)"
    ]
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(response.shape[0]):
        ax.plot(t, response[i], label=nomes[i])
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Deslocamento (m)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)
