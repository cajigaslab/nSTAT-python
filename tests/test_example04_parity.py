#!/usr/bin/env python3
"""Example 04 — Cross-language parity test.

Runs the Example 04 analysis (Place-Cell Receptive Fields) in *both*
Python and MATLAB and compares every numerical output.

This example uses precomputed FitResult structures, so all comparisons
are deterministic:
  - FitResult coefficients (b values)
  - KS statistics, AIC, BIC, logLL
  - FitSummary delta statistics (dKS, dAIC, dBIC)
  - Spatial grid design matrices (Gaussian, Zernike)
  - Place field heatmap values

Requirements:
    - MATLAB Engine API for Python (``pip install matlabengine``)
    - nSTAT MATLAB repo at ``MATLAB_NSTAT`` path below

Usage::

    python tests/test_example04_parity.py
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nstat.data_manager import ensure_example_data

MATLAB_NSTAT = Path("/Users/iahncajigas/Library/CloudStorage/Dropbox/Claude/nSTAT")
TOL = 1e-8


# ═══════════════════════════════════════════════════════════════════════════
# MATLAB side
# ═══════════════════════════════════════════════════════════════════════════
def run_matlab():
    """Run Example 04 analysis in MATLAB."""
    import matlab.engine

    py_data_dir = str(ensure_example_data(download=True))

    eng = matlab.engine.start_matlab()
    eng.addpath(str(MATLAB_NSTAT), nargout=0)
    eng.addpath(str(MATLAB_NSTAT / "libraries" / "zernike"), nargout=0)

    # ======= Load data =======
    eng.eval(f"""
        dataDir = '{py_data_dir}';
        d1 = load(fullfile(dataDir, 'Place Cells', 'PlaceCellDataAnimal1.mat'));
        d2 = load(fullfile(dataDir, 'Place Cells', 'PlaceCellDataAnimal2.mat'));
        nCells1 = length(d1.neuron);
        nCells2 = length(d2.neuron);
        nTimePoints1 = length(d1.time);
        nTimePoints2 = length(d2.time);
    """, nargout=0)

    nCells1 = int(eng.eval("nCells1"))
    nCells2 = int(eng.eval("nCells2"))
    nTimePoints1 = int(eng.eval("nTimePoints1"))
    nTimePoints2 = int(eng.eval("nTimePoints2"))

    # ======= Load FitResults =======
    eng.eval(f"""
        r1 = load(fullfile(dataDir, 'PlaceCellAnimal1Results.mat'));
        r2 = load(fullfile(dataDir, 'PlaceCellAnimal2Results.mat'));
    """, nargout=0)

    # Extract coefficients for specific cells
    eng.eval("""
        % Animal 1, cell 1 (1-indexed)
        b_a1c1_g = r1.resStruct{1}.b{1};  % Gaussian coefficients
        b_a1c1_z = r1.resStruct{1}.b{2};  % Zernike coefficients

        % Animal 1, cell 25 (example cell)
        b_a1c25_g = r1.resStruct{25}.b{1};
        b_a1c25_z = r1.resStruct{25}.b{2};

        % Animal 2, cell 1
        b_a2c1_g = r2.resStruct{1}.b{1};
        b_a2c1_z = r2.resStruct{1}.b{2};
    """, nargout=0)

    b_a1c1_g = np.array(eng.eval("b_a1c1_g")).flatten()
    b_a1c1_z = np.array(eng.eval("b_a1c1_z")).flatten()
    b_a1c25_g = np.array(eng.eval("b_a1c25_g")).flatten()
    b_a1c25_z = np.array(eng.eval("b_a1c25_z")).flatten()
    b_a2c1_g = np.array(eng.eval("b_a2c1_g")).flatten()
    b_a2c1_z = np.array(eng.eval("b_a2c1_z")).flatten()

    # ======= Extract KS, AIC, BIC directly from resStruct =======
    # (Bypasses FitResult/FitResSummary object construction —
    # all values are already stored in the serialized struct.)
    eng.eval("""
        % Extract AIC, BIC, KSStats directly from resStruct
        nModels = length(r1.resStruct{1}.AIC);
        AIC1 = zeros(nCells1, nModels);
        BIC1 = zeros(nCells1, nModels);
        KSStats1 = zeros(nCells1, nModels);
        for i = 1:nCells1
            AIC1(i,:) = r1.resStruct{i}.AIC;
            BIC1(i,:) = r1.resStruct{i}.BIC;
            KSStats1(i,:) = r1.resStruct{i}.KSStats.ks_stat;
        end

        nModels2 = length(r2.resStruct{1}.AIC);
        AIC2 = zeros(nCells2, nModels2);
        BIC2 = zeros(nCells2, nModels2);
        KSStats2 = zeros(nCells2, nModels2);
        for i = 1:nCells2
            AIC2(i,:) = r2.resStruct{i}.AIC;
            BIC2(i,:) = r2.resStruct{i}.BIC;
            KSStats2(i,:) = r2.resStruct{i}.KSStats.ks_stat;
        end

        % Delta statistics
        dKS1 = KSStats1(:,1) - KSStats1(:,2);
        dAIC1 = AIC1(:,2) - AIC1(:,1);
        dBIC1 = BIC1(:,2) - BIC1(:,1);
        dKS2 = KSStats2(:,1) - KSStats2(:,2);
        dAIC2 = AIC2(:,2) - AIC2(:,1);
        dBIC2 = BIC2(:,2) - BIC2(:,1);
    """, nargout=0)

    KSStats1 = np.array(eng.eval("KSStats1"))
    KSStats2 = np.array(eng.eval("KSStats2"))
    AIC1 = np.array(eng.eval("AIC1"))
    BIC1 = np.array(eng.eval("BIC1"))
    dKS1 = np.array(eng.eval("dKS1")).flatten()
    dAIC1 = np.array(eng.eval("dAIC1")).flatten()
    dBIC1 = np.array(eng.eval("dBIC1")).flatten()
    dKS2 = np.array(eng.eval("dKS2")).flatten()
    dAIC2 = np.array(eng.eval("dAIC2")).flatten()
    dBIC2 = np.array(eng.eval("dBIC2")).flatten()

    # ======= Spatial grid and design matrices =======
    eng.eval("""
        gridRes = 201;
        xGrid = linspace(-1, 1, gridRes);
        yGrid = linspace(-1, 1, gridRes);
        [xx, yy] = meshgrid(xGrid, yGrid);
        yy = flipud(yy);
        xx = fliplr(xx);
        % Use row-major (C-order) unravel to match Python's .ravel()
        xf = reshape(xx', [], 1);
        yf = reshape(yy', [], 1);

        % Gaussian design: [1, x, y, x^2, y^2, xy]
        gridDesignGauss = [ones(size(xf)), xf, yf, xf.^2, yf.^2, xf.*yf];

        % Zernike design: [1, z1, ..., z9]
        % Use MATLAB's zernfun or the toolbox implementation
        [theta, rho] = cart2pol(xf, yf);
        mask = rho <= 1;
        Z = zeros(length(xf), 9);
        nz = [0 1 1 2 2 2 3 3 3];
        mz = [0 -1 1 -2 0 2 -3 -1 1];
        for k = 1:9
            tmp = zeros(size(xf));
            tmp(mask) = zernfun(nz(k), mz(k), rho(mask), theta(mask), 'norm');
            Z(:,k) = tmp;
        end
        gridDesignZern = [ones(size(xf)), Z];
    """, nargout=0)

    gridDesignGauss_col0 = np.array(eng.eval("gridDesignGauss(:,1)")).flatten()
    gridDesignGauss_col3 = np.array(eng.eval("gridDesignGauss(:,4)")).flatten()
    gridDesignZern_col0 = np.array(eng.eval("gridDesignZern(:,1)")).flatten()
    gridDesignZern_col5 = np.array(eng.eval("gridDesignZern(:,5)")).flatten()

    # ======= Place field for cell 25 (example cell) =======
    eng.eval("""
        sr_ex = r1.resStruct{25}.lambda.sampleRate;
        coeffs_g = r1.resStruct{25}.b{1};
        coeffs_z = r1.resStruct{25}.b{2};
        % Compute field and reshape using row-major to match Python's reshape
        field_g_flat = exp(gridDesignGauss(:,1:length(coeffs_g)) * coeffs_g) * sr_ex;
        field_z_flat = exp(gridDesignZern(:,1:length(coeffs_z)) * coeffs_z) * sr_ex;
        field_g = reshape(field_g_flat, gridRes, gridRes)';  % transpose for row-major
        field_z = reshape(field_z_flat, gridRes, gridRes)';  % transpose for row-major
    """, nargout=0)

    sr_ex = float(eng.eval("sr_ex"))
    field_g_row0 = np.array(eng.eval("field_g(1,:)")).flatten()
    field_g_row100 = np.array(eng.eval("field_g(101,:)")).flatten()
    field_z_row0 = np.array(eng.eval("field_z(1,:)")).flatten()
    field_z_row100 = np.array(eng.eval("field_z(101,:)")).flatten()

    eng.quit()

    return {
        "nCells1": nCells1,
        "nCells2": nCells2,
        "nTimePoints1": nTimePoints1,
        "nTimePoints2": nTimePoints2,
        "b_a1c1_g": b_a1c1_g,
        "b_a1c1_z": b_a1c1_z,
        "b_a1c25_g": b_a1c25_g,
        "b_a1c25_z": b_a1c25_z,
        "b_a2c1_g": b_a2c1_g,
        "b_a2c1_z": b_a2c1_z,
        "KSStats1": KSStats1,
        "KSStats2": KSStats2,
        "AIC1": AIC1,
        "BIC1": BIC1,
        "dKS1": dKS1,
        "dAIC1": dAIC1,
        "dBIC1": dBIC1,
        "dKS2": dKS2,
        "dAIC2": dAIC2,
        "dBIC2": dBIC2,
        "gridDesignGauss_col0": gridDesignGauss_col0,
        "gridDesignGauss_col3": gridDesignGauss_col3,
        "gridDesignZern_col0": gridDesignZern_col0,
        "gridDesignZern_col5": gridDesignZern_col5,
        "sr_ex": sr_ex,
        "field_g_row0": field_g_row0,
        "field_g_row100": field_g_row100,
        "field_z_row0": field_z_row0,
        "field_z_row100": field_z_row100,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Python side
# ═══════════════════════════════════════════════════════════════════════════
def run_python():
    """Run Example 04 analysis in Python."""
    import matplotlib
    matplotlib.use("Agg")

    from scipy.io import loadmat
    from nstat import FitResult, FitSummary, TrialConfig, ConfigCollection, Covariate
    from nstat.core import nspikeTrain
    from nstat.zernike import zernike_basis_from_cartesian

    data_dir = ensure_example_data(download=True)

    # ======= Load data =======
    d1 = loadmat(str(data_dir / "Place Cells" / "PlaceCellDataAnimal1.mat"), squeeze_me=True)
    d2 = loadmat(str(data_dir / "Place Cells" / "PlaceCellDataAnimal2.mat"), squeeze_me=True)
    x1 = np.asarray(d1["x"], dtype=float).ravel()
    y1 = np.asarray(d1["y"], dtype=float).ravel()
    t1 = np.asarray(d1["time"], dtype=float).ravel()
    neurons1 = np.asarray(d1["neuron"], dtype=object).ravel()
    x2 = np.asarray(d2["x"], dtype=float).ravel()
    y2 = np.asarray(d2["y"], dtype=float).ravel()
    t2 = np.asarray(d2["time"], dtype=float).ravel()
    neurons2 = np.asarray(d2["neuron"], dtype=object).ravel()

    nCells1 = len(neurons1)
    nCells2 = len(neurons2)

    # ======= Load FitResults =======
    # Use the same loader as example04
    sys.path.insert(0, str(REPO_ROOT / "examples" / "paper"))
    from example04_place_cells_continuous_stimulus import _load_animal_results

    fitResults1 = _load_animal_results(
        data_dir / "PlaceCellAnimal1Results.mat", x1, y1, t1, neurons1)
    fitResults2 = _load_animal_results(
        data_dir / "PlaceCellAnimal2Results.mat", x2, y2, t2, neurons2)

    # Extract coefficients
    b_a1c1_g = np.asarray(fitResults1[0].b[0], dtype=float).ravel()
    b_a1c1_z = np.asarray(fitResults1[0].b[1], dtype=float).ravel()
    b_a1c25_g = np.asarray(fitResults1[24].b[0], dtype=float).ravel()
    b_a1c25_z = np.asarray(fitResults1[24].b[1], dtype=float).ravel()
    b_a2c1_g = np.asarray(fitResults2[0].b[0], dtype=float).ravel()
    b_a2c1_z = np.asarray(fitResults2[0].b[1], dtype=float).ravel()

    # ======= FitSummary statistics =======
    summary1 = FitSummary(fitResults1)
    summary2 = FitSummary(fitResults2)

    KSStats1 = np.asarray(summary1.KSStats, dtype=float)
    AIC1 = np.asarray(summary1.AIC, dtype=float)
    BIC1 = np.asarray(summary1.BIC, dtype=float)

    dKS1 = KSStats1[:, 0] - KSStats1[:, 1]
    dAIC1 = AIC1[:, 1] - AIC1[:, 0]
    dBIC1 = BIC1[:, 1] - BIC1[:, 0]

    KSStats2 = np.asarray(summary2.KSStats, dtype=float)
    AIC2 = np.asarray(summary2.AIC, dtype=float)
    BIC2 = np.asarray(summary2.BIC, dtype=float)

    dKS2 = KSStats2[:, 0] - KSStats2[:, 1]
    dAIC2 = AIC2[:, 1] - AIC2[:, 0]
    dBIC2 = BIC2[:, 1] - BIC2[:, 0]

    # ======= Spatial grid and design matrices =======
    grid_res = 201
    xGrid = np.linspace(-1, 1, grid_res)
    yGrid = np.linspace(-1, 1, grid_res)
    xx, yy = np.meshgrid(xGrid, yGrid)
    yy = np.flipud(yy)
    xx = np.fliplr(xx)
    xf, yf = xx.ravel(), yy.ravel()

    # Gaussian design
    gridDesignGauss = np.column_stack([
        np.ones(xf.size), xf, yf, xf**2, yf**2, xf * yf
    ])

    # Zernike design
    zBasis = zernike_basis_from_cartesian(xf, yf, fill_value=0.0)
    gridDesignZern = np.column_stack([np.ones(xf.size), zBasis])

    # ======= Place field for cell 25 =======
    sr_ex = float(fitResults1[24].lambda_signal.sampleRate)
    coeffs_g = b_a1c25_g
    coeffs_z = b_a1c25_z
    field_g = np.exp(gridDesignGauss[:, :coeffs_g.size] @ coeffs_g).reshape(grid_res, grid_res) * sr_ex
    field_z = np.exp(gridDesignZern[:, :coeffs_z.size] @ coeffs_z).reshape(grid_res, grid_res) * sr_ex

    return {
        "nCells1": nCells1,
        "nCells2": nCells2,
        "nTimePoints1": len(t1),
        "nTimePoints2": len(t2),
        "b_a1c1_g": b_a1c1_g,
        "b_a1c1_z": b_a1c1_z,
        "b_a1c25_g": b_a1c25_g,
        "b_a1c25_z": b_a1c25_z,
        "b_a2c1_g": b_a2c1_g,
        "b_a2c1_z": b_a2c1_z,
        "KSStats1": KSStats1,
        "KSStats2": KSStats2,
        "AIC1": AIC1,
        "BIC1": BIC1,
        "dKS1": dKS1,
        "dAIC1": dAIC1,
        "dBIC1": dBIC1,
        "dKS2": dKS2,
        "dAIC2": dAIC2,
        "dBIC2": dBIC2,
        "gridDesignGauss_col0": gridDesignGauss[:, 0],
        "gridDesignGauss_col3": gridDesignGauss[:, 3],
        "gridDesignZern_col0": gridDesignZern[:, 0],
        "gridDesignZern_col5": gridDesignZern[:, 4],
        "sr_ex": sr_ex,
        "field_g_row0": field_g[0, :],
        "field_g_row100": field_g[100, :],
        "field_z_row0": field_z[0, :],
        "field_z_row100": field_z[100, :],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Compare
# ═══════════════════════════════════════════════════════════════════════════
def compare(py: dict, ml: dict):
    """Compare Python and MATLAB results; return (passed, total)."""
    passed = 0
    total = 0

    def check(name, py_val, ml_val, tol=TOL):
        nonlocal passed, total
        total += 1
        py_a = np.asarray(py_val, dtype=float).flatten()
        ml_a = np.asarray(ml_val, dtype=float).flatten()
        if py_a.shape != ml_a.shape:
            print(f"  ✗ {name}  SHAPE MISMATCH py={py_a.shape} ml={ml_a.shape}")
            return
        if py_a.size == 0 and ml_a.size == 0:
            print(f"  ✓ {name}  (both empty)")
            passed += 1
            return
        delta = np.max(np.abs(py_a - ml_a))
        ok = delta < tol
        sym = "✓" if ok else "✗"
        extra = ""
        if not ok:
            idx = int(np.argmax(np.abs(py_a - ml_a)))
            extra = f" at [{idx}] py={py_a[idx]:.6f} ml={ml_a[idx]:.6f}"
        print(f"  {sym} {name}  (max Δ = {delta:.2e}){extra}")
        if ok:
            passed += 1

    def check_int(name, py_val, ml_val):
        nonlocal passed, total
        total += 1
        ok = py_val == ml_val
        sym = "✓" if ok else "✗"
        print(f"  {sym} {name}  py={py_val} ml={ml_val}")
        if ok:
            passed += 1

    # ─────── Data Loading ───────
    print("\n═══ Data Loading ═══")
    check_int("nCells1", py["nCells1"], ml["nCells1"])
    check_int("nCells2", py["nCells2"], ml["nCells2"])
    check_int("nTimePoints1", py["nTimePoints1"], ml["nTimePoints1"])
    check_int("nTimePoints2", py["nTimePoints2"], ml["nTimePoints2"])

    # ─────── FitResult Coefficients ───────
    print("\n═══ FitResult Coefficients ═══")
    check("Animal1 Cell1 Gaussian b", py["b_a1c1_g"], ml["b_a1c1_g"])
    check("Animal1 Cell1 Zernike b", py["b_a1c1_z"], ml["b_a1c1_z"])
    check("Animal1 Cell25 Gaussian b", py["b_a1c25_g"], ml["b_a1c25_g"])
    check("Animal1 Cell25 Zernike b", py["b_a1c25_z"], ml["b_a1c25_z"])
    check("Animal2 Cell1 Gaussian b", py["b_a2c1_g"], ml["b_a2c1_g"])
    check("Animal2 Cell1 Zernike b", py["b_a2c1_z"], ml["b_a2c1_z"])

    # ─────── FitSummary Statistics ───────
    print("\n═══ FitSummary Statistics ═══")
    check("KSStats1", py["KSStats1"], ml["KSStats1"])
    check("KSStats2", py["KSStats2"], ml["KSStats2"])
    check("AIC1", py["AIC1"], ml["AIC1"])
    check("BIC1", py["BIC1"], ml["BIC1"])
    check("dKS1 (Gauss-Zern)", py["dKS1"], ml["dKS1"])
    check("dAIC1 (Zern-Gauss)", py["dAIC1"], ml["dAIC1"])
    check("dBIC1 (Zern-Gauss)", py["dBIC1"], ml["dBIC1"])
    check("dKS2 (Gauss-Zern)", py["dKS2"], ml["dKS2"])
    check("dAIC2 (Zern-Gauss)", py["dAIC2"], ml["dAIC2"])
    check("dBIC2 (Zern-Gauss)", py["dBIC2"], ml["dBIC2"])

    # ─────── Design Matrices ───────
    print("\n═══ Design Matrices ═══")
    check("Gaussian design col0", py["gridDesignGauss_col0"], ml["gridDesignGauss_col0"])
    check("Gaussian design col3 (x²)", py["gridDesignGauss_col3"], ml["gridDesignGauss_col3"])
    check("Zernike design col0", py["gridDesignZern_col0"], ml["gridDesignZern_col0"])
    check("Zernike design col5", py["gridDesignZern_col5"], ml["gridDesignZern_col5"])

    # ─────── Place Fields ───────
    # sampleRate: Python's Covariate computes 1/dt from time vector (FP rounding),
    # MATLAB stores the exact value.  ~0.02% difference → propagates to fields.
    print("\n═══ Place Field Computation ═══")
    check("sampleRate (cell 25)", py["sr_ex"], ml["sr_ex"], tol=1e-3)
    check("Gaussian field row0", py["field_g_row0"], ml["field_g_row0"], tol=1e-3)
    check("Gaussian field row100", py["field_g_row100"], ml["field_g_row100"], tol=1e-3)
    check("Zernike field row0", py["field_z_row0"], ml["field_z_row0"], tol=1e-3)
    check("Zernike field row100", py["field_z_row100"], ml["field_z_row100"], tol=1e-3)

    return passed, total


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = _time.time()
    print("Starting MATLAB engine …")
    ml = run_matlab()
    t_ml = _time.time() - t0
    print(f"  MATLAB done in {t_ml:.1f}s")

    t1 = _time.time()
    print("\nRunning Example 04 in Python …")
    py = run_python()
    t_py = _time.time() - t1
    print(f"  Python done in {t_py:.1f}s")

    passed, total = compare(py, ml)

    print(f"\n{'═' * 60}")
    status = "✓ ALL PASS" if passed == total else f"✗ {total - passed} FAILED"
    print(f"  EXAMPLE 04 PARITY: {passed}/{total} passed  {status}")
    print(f"  MATLAB: {t_ml:.1f}s | Python: {t_py:.1f}s")
    print(f"{'═' * 60}")

    sys.exit(0 if passed == total else 1)
