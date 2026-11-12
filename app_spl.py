import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="SPL Metode Cramer", layout="wide")
st.title("Penyelesaian SPL dengan Metode Cramer")
st.write("Unggah file CSV/Excel berisi koefisien dan hasil setiap persamaan, "
         "lalu pilih kolom variabel (x) dan kolom hasil (b).")


def baca_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_excel(file)


def bangun_matriks(df, kolom_fitur, kolom_target):
    data = df[kolom_fitur + [kolom_target]].apply(pd.to_numeric, errors="raise")
    A = data[kolom_fitur].to_numpy(dtype=float)
    b = data[kolom_target].to_numpy(dtype=float)
    return A, b


def selesaikan_cramer(A, b):
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError(
            f"Metode Cramer hanya untuk matriks persegi. "
            f"Jumlah persamaan = {n}, jumlah variabel = {A.shape[1]}."
        )

    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-12:
        raise ValueError("Determinan A = 0, sistem tidak punya solusi tunggal.")

    det_i = []
    x = []
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b              # kolom ke-i diganti vektor b
        d = np.linalg.det(Ai)
        det_i.append(d)
        x.append(d / det_A)

    return np.array(x), det_A, det_i


file = st.file_uploader("Unggah file data (CSV / Excel)", type=["csv", "xlsx", "xls"])

if file is None:
    st.info("Silakan unggah file data terlebih dahulu.")
else:
    df = baca_file(file)
    st.subheader("Data yang diunggah")
    st.dataframe(df)

    kolom_numerik = df.select_dtypes(include="number").columns.tolist()
    kolom_target = st.selectbox(
        "Kolom hasil (b)", kolom_numerik, index=len(kolom_numerik) - 1
    )

    kolom_lain = [k for k in kolom_numerik if k != kolom_target]
    default_fitur = [k for k in kolom_lain
                     if k.lower() not in ("sampel", "no", "id")]
    kolom_fitur = st.multiselect("Kolom variabel (x)", kolom_lain,
                                 default=default_fitur)

    if st.button("Selesaikan", type="primary"):
        try:
            if len(kolom_fitur) == 0:
                raise ValueError("Pilih minimal satu kolom variabel.")
            A, b = bangun_matriks(df, kolom_fitur, kolom_target)
            x, det_A, det_i = selesaikan_cramer(A, b)
            st.session_state["hasil"] = dict(
                A=A, b=b, x=x, det_A=det_A, det_i=det_i, nama=kolom_fitur
            )
        except Exception as e:
            st.session_state.pop("hasil", None)
            st.error(f"Terjadi kesalahan: {e}")

    if "hasil" in st.session_state:
        h = st.session_state["hasil"]
        nama = h["nama"]

        st.subheader("Matriks koefisien A")
        st.dataframe(pd.DataFrame(h["A"], columns=nama))

        st.subheader("Vektor b")
        st.dataframe(pd.DataFrame({kolom_target: h["b"]}))

        st.subheader("Determinan")
        tabel_det = {"Determinan": ["det(A)"] + [f"det(A{i+1}) → {n}" for i, n in enumerate(nama)],
                     "Nilai": [round(h["det_A"], 4)] + [round(d, 4) for d in h["det_i"]]}
        st.dataframe(pd.DataFrame(tabel_det))

        st.subheader("Solusi (x = det(Ai) / det(A))")
        st.dataframe(pd.DataFrame({"variabel": nama, "nilai": np.round(h["x"], 4)}))

        # Verifikasi: A @ x harus sama dengan b
        st.subheader("Verifikasi A·x = b")
        hasil_cek = h["A"] @ h["x"]
        st.dataframe(pd.DataFrame({
            "A·x": np.round(hasil_cek, 4),
            "b": h["b"],
            "selisih": np.round(hasil_cek - h["b"], 6),
        }))
        if np.allclose(hasil_cek, h["b"]):
            st.success("Solusi benar: A·x sama dengan b.")