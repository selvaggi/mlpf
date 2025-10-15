import numpy as np
import math

import awkward as ak

def build_dummy_array(num, dtype=np.int64):
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.zeros(num + 1, dtype=np.int64)),
            ak.from_numpy(np.array([], dtype=dtype), highlevel=False),
        )
    )

def _concat_records(table):
    table1 =  {k : ak.from_iter([record[k][event] for record in table for event in range(len(record[k])) ]) for k in table[0].fields}
    for k in table1.keys():
            if len(ak.flatten(table1[k])) == 0:
                table1[k] = build_dummy_array(len(table1[k]), np.float32)
    table1 = ak.Record(table1)
    return table1

def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return ak.concatenate(arrays, axis=axis)


def _stack(arrays, axis=1):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.stack(arrays, axis=axis)
    else:
        return ak.concatenate(arrays, axis=axis)


def _pad_vector(a, value=-1, dtype="float32"):
    maxlen = 2000
    maxlen2 = 5

    x = (np.ones((len(a), maxlen, maxlen2)) * value).astype(dtype)
    for idx, s in enumerate(a):
        for idx_vec, s_vec in enumerate(s):
            x[idx, idx_vec, : len(s_vec)] = s_vec
    return x


def _pad(a, maxlen, value=0, dtype="float32"):
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    elif isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    else:
        x = (np.ones((len(a), maxlen)) * value).astype(dtype)
        for idx, s in enumerate(a):
            if not len(s):
                continue
            trunc = s[:maxlen].astype(dtype)
            x[idx, : len(trunc)] = trunc
        return x


def _repeat_pad(a, maxlen, shuffle=False, dtype="float32"):
    x = ak.to_numpy(ak.flatten(a))
    x = np.tile(x, int(np.ceil(len(a) * maxlen / len(x))))
    if shuffle:
        np.random.shuffle(x)
    x = x[: len(a) * maxlen].reshape((len(a), maxlen))
    mask = _pad(ak.zeros_like(a), maxlen, value=1)
    x = _pad(a, maxlen) + mask * x
    return ak.values_astype(x, dtype)


def _clip(a, a_min, a_max):
    try:
        return np.clip(a, a_min, a_max)
    except ValueError:
        return ak.unflatten(np.clip(ak.flatten(a), a_min, a_max), ak.num(a))


def _knn(support, query, k, n_jobs=1):
    from scipy.spatial import cKDTree

    kdtree = cKDTree(support)
    d, idx = kdtree.query(query, k, n_jobs=n_jobs)
    return idx


def _batch_knn(supports, queries, k, maxlen_s, maxlen_q=None, n_jobs=1):
    assert len(supports) == len(queries)
    if maxlen_q is None:
        maxlen_q = maxlen_s
    batch_knn_idx = np.ones((len(supports), maxlen_q, k), dtype="int32") * (
        maxlen_s - 1
    )
    for i, (s, q) in enumerate(zip(supports, queries)):
        batch_knn_idx[i, : len(q[:maxlen_q]), :] = _knn(
            s[:maxlen_s], q[:maxlen_q], k, n_jobs=n_jobs
        ).reshape(
            (-1, k)
        )  # (len(q), k)
    return batch_knn_idx


def _batch_permute_indices(array, maxlen):
    batch_permute_idx = np.tile(np.arange(maxlen), (len(array), 1))
    for i, a in enumerate(array):
        batch_permute_idx[i, : len(a)] = np.random.permutation(len(a[:maxlen]))
    return batch_permute_idx


def _batch_argsort(array, maxlen):
    batch_argsort_idx = np.tile(np.arange(maxlen), (len(array), 1))
    for i, a in enumerate(array):
        batch_argsort_idx[i, : len(a)] = np.argsort(a[:maxlen])
    return batch_argsort_idx


def _batch_gather(array, indices):
    out = array.zeros_like()
    for i, (a, idx) in enumerate(zip(array, indices)):
        maxlen = min(len(a), len(idx))
        out[i][:maxlen] = a[idx[:maxlen]]
    return out


def _p4_from_pxpypze(px, py, pz, energy):
    import vector

    vector.register_awkward()
    return vector.zip({"px": px, "py": py, "pz": pz, "energy": energy})


def _p4_from_ptetaphie(pt, eta, phi, energy):
    import vector

    vector.register_awkward()
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "energy": energy})


def _p4_from_ptetaphim(pt, eta, phi, mass):
    import vector

    vector.register_awkward()
    return vector.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})


def _get_variable_names(expr, exclude=["awkward", "ak", "np", "numpy", "math"]):
    import ast

    root = ast.parse(expr)
    return sorted(
        {
            node.id
            for node in ast.walk(root)
            if isinstance(node, ast.Name) and not node.id.startswith("_")
        }
        - set(exclude)
    )


def _eval_expr(expr, table):
    tmp = {k: table[k] for k in _get_variable_names(expr)}
    tmp.update(
        {
            "math": math,
            "np": np,
            "numpy": np,
            "ak": ak,
            "awkward": ak,
            "_concat": _concat,
            "_stack": _stack,
            "_pad": _pad,
            "_repeat_pad": _repeat_pad,
            "_clip": _clip,
            "_batch_knn": _batch_knn,
            "_batch_permute_indices": _batch_permute_indices,
            "_batch_argsort": _batch_argsort,
            "_batch_gather": _batch_gather,
            "_p4_from_pxpypze": _p4_from_pxpypze,
            "_p4_from_ptetaphie": _p4_from_ptetaphie,
            "_p4_from_ptetaphim": _p4_from_ptetaphim,
        }
    )
    return eval(expr, tmp)
