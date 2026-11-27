import numpy as np, pandas as pd
import pennylane as qml
from sklearn.decomposition import PCA
from ablation_vqc import _vqc, _fit_vqc, _m

# filtro cuántico 2x2 (4 qubits), pesos fijos como el QCK de la rama
_dev4 = qml.device("default.qubit", wires=4)
_rng = np.random.default_rng(42)
_qw = _rng.uniform(-np.pi, np.pi, (2, 4, 2))

@qml.qnode(_dev4)
def _kernel(x):
    for i in range(4):
        qml.RY(np.pi * x[i], wires=i)
        qml.RX(np.pi * x[i] / 2, wires=i)
    for l in range(2):
        for i in range(4):
            qml.RX(_qw[l, i, 0], wires=i); qml.RY(_qw[l, i, 1], wires=i)
        qml.CNOT([0, 1]); qml.CNOT([1, 2]); qml.CNOT([2, 3]); qml.CNOT([3, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

def _quanv(amp64):
    # desliza el filtro sobre el downsample 8x8 y promedia cada canal -> 4 features
    img = np.asarray(amp64, float).reshape(8, 8)
    img = (img - img.min()) / (np.ptp(img) + 1e-8)
    ch = [[], [], [], []]
    for r in range(0, 7, 2):
        for c in range(0, 7, 2):
            z = _kernel(img[r:r+2, c:c+2].ravel())
            for i in range(4): ch[i].append(float(z[i]))
    return np.array([np.mean(ch[i]) for i in range(4)])


def run_5_16(X_all, X_amp, meta, idx, y, VIDEO, SELECTED,
             AngleScaler, macro_ba, video_weights, train_svm,
             method="Crudos", seeds=(17, 42, 101), epochs=40, layers=4):
    tr, va, te = idx["train"], idx["val"], idx["test"]
    sel8, Xf = list(SELECTED[method]), X_all[method]

    hand = Xf[:, sel8]                                     # 8 features clásicas
    qf = np.stack([_quanv(a) for a in X_amp[method]])      # extractor cuántico -> 4
    both = np.column_stack([hand, qf])                     # ambas
    d = qf.shape[1]
    pca = PCA(n_components=d, random_state=0).fit(X_amp[method][tr])
    ctrl = pca.transform(X_amp[method])                    # control clásico, misma dim

    entradas = [("8 features", hand), ("circuito", qf), ("ambas", both), ("control PCA", ctrl)]
    rows = []
    for nombre, Xraw in entradas:
        sc = AngleScaler().fit(Xraw[tr]); Xin = sc(Xraw); nq = Xin.shape[1]
        svm = train_svm(Xin[tr], y[tr], VIDEO[tr], Xin[va], y[va], VIDEO[va])
        sp = svm.predict(Xin[te])
        rows.append(dict(entrada=nombre, dim=nq, modelo="SVM",
                         bal_video=round(macro_ba(y[te], sp, VIDEO[te]), 3), **{k: round(v, 3) for k, v in _m(y[te], sp).items()}))
        circuit, _ = _vqc(nq, layers, "angle")
        acc = []
        for s in seeds:
            P, thr, proba = _fit_vqc(circuit, nq, layers, "angle", Xin[tr], y[tr], VIDEO[tr],
                                     Xin[va], y[va], VIDEO[va], macro_ba, video_weights, s, epochs)
            pr = (proba(Xin[te], P) >= thr).astype(int)
            acc.append(dict(bal=macro_ba(y[te], pr, VIDEO[te]), **_m(y[te], pr)))
        v = pd.DataFrame(acc).mean()
        rows.append(dict(entrada=nombre, dim=nq, modelo="VQC",
                         bal_video=round(float(v["bal"]), 3), ba=round(float(v["ba"]), 3), f1=round(float(v["f1"]), 3)))
    return pd.DataFrame(rows)
