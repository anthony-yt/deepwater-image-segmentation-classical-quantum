import numpy as np, pandas as pd, time
import torch, pennylane as qml
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, f1_score

MAX_QUBITS = 16   # tope del simulador: 2^n amplitudes


def _vqc(nq, nlayers, enc):
    dev = qml.device("default.qubit", wires=nq)
    w = list(range(nq))
    if enc == "angle":
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def c(x, W):
            for l in range(nlayers):
                qml.AngleEmbedding(x, wires=w, rotation="Y")   # re-uploading
                qml.StronglyEntanglingLayers(W[l:l+1], wires=w)
            return [qml.expval(qml.PauliZ(i)) for i in w]
        return c, nq
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def c(x, W):
        qml.AmplitudeEmbedding(x, wires=w, normalize=True)
        qml.StronglyEntanglingLayers(W, wires=w)
        return qml.expval(qml.PauliZ(0))
    return c, 1


def _fit_vqc(circuit, nq, nlayers, enc, xtr, ytr, vtr, xva, yva, vva,
             macro_ba, video_weights, seed, epochs=40, patience=10, bs=64, lr=.03):
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(seed)
    W = torch.nn.Parameter(.1 * torch.randn(nlayers, nq, 3, generator=g))
    P = {"W": W}
    if enc == "angle":
        P["head"] = torch.nn.Parameter(.1 * torch.randn(nq, generator=g))
        P["b"] = torch.nn.Parameter(torch.zeros(()))

    def logit(x, P):
        if enc == "angle":
            z = torch.stack(circuit(x, P["W"]), -1)
            return z @ P["head"] + P["b"]
        p = ((1 + circuit(x, P["W"])) / 2).clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def proba(x, P):
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(logit(torch.tensor(x[i:i+256]), P)).numpy().reshape(-1)
                                   for i in range(0, len(x), 256)])

    opt = torch.optim.Adam(P.values(), lr=lr)
    xx, yy = torch.tensor(xtr), torch.tensor(ytr, dtype=torch.float64)
    ww = torch.tensor(video_weights(vtr, ytr))                 # peso por (video, clase)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    gb = torch.Generator().manual_seed(seed)
    best, bestP, wait = -1, None, 0
    for _ in range(epochs):
        for b in torch.randperm(len(xx), generator=gb).split(bs):
            opt.zero_grad()
            ((bce(logit(xx[b], P), yy[b]) * ww[b]).mean()).backward(); opt.step()
        f = macro_ba(yva, proba(xva, P) >= .5, vva)            # selección en validación
        if f > best: best, wait, bestP = f, 0, {k: v.detach().clone() for k, v in P.items()}
        else:
            wait += 1
            if wait >= patience: break
    pv = proba(xva, bestP)
    ts = np.linspace(.15, .85, 29)                             # umbral en validación
    thr = float(ts[np.argmax([macro_ba(yva, pv >= t, vva) for t in ts])])
    return bestP, thr, proba


def _m(y, p):
    return dict(ba=balanced_accuracy_score(y, p), f1=f1_score(y, p, zero_division=0))


def run_ablation(X_all, X_amp, meta, idx, y, VIDEO, CANDIDATES, SELECTED,
                 AngleScaler, macro_ba, video_weights, train_svm,
                 method="Crudos", seeds=(17, 42, 101), label_tag="SAM", epochs=40, ranking=None):
    tr, va, te = idx["train"], idx["val"], idx["test"]
    Xf, Xa, sel8 = X_all[method], X_amp[method], list(SELECTED[method])
    if ranking is None:
        mi = mutual_info_classif(Xf[tr], y[tr], random_state=0)
        ranking = list(np.argsort(mi)[::-1])                   # ranking estable de features
    def topk(k):
        return (sel8 + [c for c in ranking if c not in sel8])[:k]

    # baseline v3 + una variable cambiada por fila
    grid = [
        ("codificación", "angle-8f-4c", dict(enc="angle", cols=sel8, L=4)),
        ("codificación", "amp-6q-2c",   dict(enc="amplitude", cols=None, L=2)),
        ("capas", "angle-8f-1c", dict(enc="angle", cols=sel8, L=1)),
        ("capas", "angle-8f-2c", dict(enc="angle", cols=sel8, L=2)),
        ("capas", "angle-8f-6c", dict(enc="angle", cols=sel8, L=6)),
        ("features", "angle-4f-4c",  dict(enc="angle", cols=topk(4),  L=4)),
        ("features", "angle-12f-4c", dict(enc="angle", cols=topk(12), L=4)),
    ]

    rows = []
    for var, tag, cfg in grid:
        enc, cols, L = cfg["enc"], cfg["cols"], cfg["L"]
        if enc == "amplitude":
            Xin, nq = Xa, 6
        else:
            sc = AngleScaler().fit(Xf[tr][:, cols]); Xin, nq = sc(Xf[:, cols]), len(cols)

        if nq > MAX_QUBITS:                                    # inviable en simulador
            svm = train_svm(Xin[tr], y[tr], VIDEO[tr], Xin[va], y[va], VIDEO[va])
            sp = svm.predict(Xin[te])
            rows += [dict(etiquetas=label_tag, var=var, config=tag, modelo="VQC", ba=np.nan, bal_video=np.nan, f1=np.nan),
                     dict(etiquetas=label_tag, var=var, config=tag, modelo="SVM",
                          bal_video=round(macro_ba(y[te], sp, VIDEO[te]), 3), **{k: round(v, 3) for k, v in _m(y[te], sp).items()})]
            continue

        circuit, _ = _vqc(nq, L, enc)
        acc = []; t0 = time.perf_counter()
        for s in seeds:
            P, thr, proba = _fit_vqc(circuit, nq, L, enc, Xin[tr], y[tr], VIDEO[tr],
                                     Xin[va], y[va], VIDEO[va], macro_ba, video_weights, s, epochs)
            pr = (proba(Xin[te], P) >= thr).astype(int)
            acc.append(dict(bal=macro_ba(y[te], pr, VIDEO[te]), **_m(y[te], pr)))
        v = pd.DataFrame(acc).mean(); sec = (time.perf_counter() - t0) / len(seeds)

        svm = train_svm(Xin[tr], y[tr], VIDEO[tr], Xin[va], y[va], VIDEO[va])
        sp = svm.predict(Xin[te])
        rows += [dict(etiquetas=label_tag, var=var, config=tag, modelo="VQC",
                      ba=round(float(v["ba"]), 3), bal_video=round(float(v["bal"]), 3), f1=round(float(v["f1"]), 3), seg=round(sec, 1)),
                 dict(etiquetas=label_tag, var=var, config=tag, modelo="SVM",
                      bal_video=round(macro_ba(y[te], sp, VIDEO[te]), 3), **{k: round(v, 3) for k, v in _m(y[te], sp).items()})]

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="config", columns="modelo", values="bal_video")
    piv["VQC_vs_SVM"] = (piv["VQC"] - piv["SVM"]).round(3)     # cuánto aporta el VQC sobre el SVM
    return df.merge(piv["VQC_vs_SVM"].reset_index(), on="config", how="left")
