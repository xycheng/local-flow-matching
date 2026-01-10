import numpy as np
import datasets_tabular as dt


def stats(x: np.ndarray):
    return {
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "max": x.max(),
        "q01": np.quantile(x, 0.01),
        "q99": np.quantile(x, 0.99),
    }


def check_one(name, data_root="./data"):
    print(f"\n{name.upper()}")

    ds = dt.get_tabular_dataset(name, data_root=data_root)

    xtr = ds.trn.x
    xva = ds.val.x
    xte = ds.tst.x

    print(f"  train: {xtr.shape}")
    print(f"  val  : {xva.shape}")
    print(f"  test : {xte.shape}")
    print(f"  total: {xtr.shape[0] + xva.shape[0] + xte.shape[0]}")
    print(f"  dim  : {xtr.shape[1]}")

    s = stats(xtr)
    print("  train stats:")
    print(f"    mean : {s['mean']:.4f}")
    print(f"    std  : {s['std']:.4f}")
    print(f"    min  : {s['min']:.4f}")
    print(f"    max  : {s['max']:.4f}")
    print(f"    q01  : {s['q01']:.4f}")
    print(f"    q99  : {s['q99']:.4f}")


if __name__ == "__main__":
    for name in ["bsds300", "gas", "miniboone", "power"]:
        check_one(name)
