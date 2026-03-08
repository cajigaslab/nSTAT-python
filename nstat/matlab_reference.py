from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_matlab_repo() -> Path:
    return _repo_root().parent / "nSTAT"


def matlab_engine_available() -> bool:
    try:
        import matlab.engine  # type: ignore
    except Exception:
        return False
    return True


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        if hasattr(value, "_data") and hasattr(value, "size"):
            return np.asarray(value._data, dtype=float).reshape(value.size, order="F")
        raise


@lru_cache(maxsize=1)
def start_matlab_engine():
    if not matlab_engine_available():
        raise RuntimeError("MATLAB Engine for Python is not available in this environment")
    import matlab.engine  # type: ignore

    return matlab.engine.start_matlab()


def _add_repo_to_path(engine, matlab_repo: Path) -> None:
    engine.addpath(str(matlab_repo), nargout=0)
    engine.addpath(str(matlab_repo / "helpfiles"), nargout=0)


def run_point_process_reference(*, matlab_repo: str | Path | None = None, seed: int = 5) -> dict[str, np.ndarray]:
    repo = Path(matlab_repo) if matlab_repo is not None else _default_matlab_repo()
    if not repo.exists():
        raise FileNotFoundError(f"MATLAB reference repo not found at {repo}")
    engine = start_matlab_engine()
    _add_repo_to_path(engine, repo)
    engine.eval(
        f"""
        rng({int(seed)});
        Ts=.001; tMin=0; tMax=50; t=tMin:Ts:tMax;
        mu=-3;
        H=tf([-1 -2 -4],[1],Ts,'Variable','z^-1');
        S=tf([1],1,Ts,'Variable','z^-1');
        E=tf([0],1,Ts,'Variable','z^-1');
        u=sin(2*pi*1*t)';
        e=zeros(length(t),1);
        stim=Covariate(t',u,'Stimulus','time','s','Voltage',{'sin'});
        ens=Covariate(t',e,'Ensemble','time','s','Spikes',{'n1'});
        [sC, lambda] = CIF.simulateCIF(mu,H,S,E,stim,ens,5,'binomial');
        ppSpikeCounts = zeros(1, sC.numSpikeTrains);
        for i=1:sC.numSpikeTrains
            ppSpikeCounts(i) = length(sC.getNST(i).spikeTimes);
        end
        ppLambdaHead = lambda.data(1:10,1)';
        """,
        nargout=0,
    )
    return {
        "spike_counts": _to_numpy(engine.workspace["ppSpikeCounts"]).reshape(-1),
        "lambda_head": _to_numpy(engine.workspace["ppLambdaHead"]).reshape(-1),
    }


def run_simulated_network_reference(*, matlab_repo: str | Path | None = None, seed: int = 4) -> dict[str, np.ndarray]:
    repo = Path(matlab_repo) if matlab_repo is not None else _default_matlab_repo()
    if not repo.exists():
        raise FileNotFoundError(f"MATLAB reference repo not found at {repo}")
    engine = start_matlab_engine()
    _add_repo_to_path(engine, repo)
    engine.eval(
        f"""
        rng({int(seed)});
        Ts=.001; tMin=0; tMax=50; t=tMin:Ts:tMax;
        mu{1}=-3; mu{2}=-3;
        H{1}=tf([-4 -2 -1],[1],Ts,'Variable','z^-1');
        H{2}=tf([-4 -2 -1],[1],Ts,'Variable','z^-1');
        S{1}=tf([1],1,Ts,'Variable','z^-1');
        S{2}=tf([-1],1,Ts,'Variable','z^-1');
        E{1}=tf([1],1,Ts,'Variable','z^-1');
        E{2}=tf([-4],1,Ts,'Variable','z^-1');
        u = sin(2*pi*1*t)';
        stim=Covariate(t',u,'Stimulus','time','s','Voltage',{'sin'});
        assignin('base','S1',S{1}); assignin('base','H1',H{1}); assignin('base','E1',E{1}); assignin('base','mu1',mu{1});
        assignin('base','S2',S{2}); assignin('base','H2',H{2}); assignin('base','E2',E{2}); assignin('base','mu2',mu{2});
        options = simget;
        [tout,~,yout] = sim('SimulatedNetwork2',[stim.minTime stim.maxTime],options,stim.dataToStructure);
        [h1Num, ~] = tfdata(H{1}, 'v');
        [h2Num, ~] = tfdata(H{2}, 'v');
        [s1Num, ~] = tfdata(S{1}, 'v');
        [s2Num, ~] = tfdata(S{2}, 'v');
        [e1Num, ~] = tfdata(E{1}, 'v');
        [e2Num, ~] = tfdata(E{2}, 'v');
        stateMat = yout(:,1:2);
        probMat = zeros(size(stateMat));
        for n = 1:size(stateMat, 1)
            hist1 = 0; hist2 = 0;
            for lag = 1:length(h1Num)
                if n-lag >= 1
                    hist1 = hist1 + h1Num(lag) * stateMat(n-lag,1);
                    hist2 = hist2 + h2Num(lag) * stateMat(n-lag,2);
                end
            end
            ens1 = 0; ens2 = 0;
            if n > 1
                ens1 = e1Num(1) * stateMat(n-1,2);
                ens2 = e2Num(1) * stateMat(n-1,1);
            end
            eta1 = mu{1} + hist1 + s1Num(1) * u(n) + ens1;
            eta2 = mu{2} + hist2 + s2Num(1) * u(n) + ens2;
            probMat(n,1) = exp(eta1) / (1 + exp(eta1));
            probMat(n,2) = exp(eta2) / (1 + exp(eta2));
        end
        netSpikeCounts = [sum(yout(:,1)>.5), sum(yout(:,2)>.5)];
        netProbHead = probMat(1:5,:);
        netStateHead = yout(1:5,1:2);
        netActual = [0 1; -4 0];
        """,
        nargout=0,
    )
    return {
        "spike_counts": _to_numpy(engine.workspace["netSpikeCounts"]).reshape(-1),
        "prob_head": _to_numpy(engine.workspace["netProbHead"]),
        "state_head": _to_numpy(engine.workspace["netStateHead"]),
        "actual_network": _to_numpy(engine.workspace["netActual"]),
    }


def run_analysis_reference(*, matlab_repo: str | Path | None = None) -> dict[str, np.ndarray]:
    repo = Path(matlab_repo) if matlab_repo is not None else _default_matlab_repo()
    if not repo.exists():
        raise FileNotFoundError(f"MATLAB reference repo not found at {repo}")
    engine = start_matlab_engine()
    _add_repo_to_path(engine, repo)
    engine.eval(
        """
        t = (0:0.1:1.0)';
        stim = Covariate(t, sin(2*pi*t), 'Stimulus', 'time', 's', '', {'stim'});
        spikeTrain = nspikeTrain([0.1 0.4 0.7], '1', 0.1, 0.0, 1.0, 'time', 's', '', '', -1);
        trial = Trial(nstColl({spikeTrain}), CovColl({stim}));
        cfg = TrialConfig({{'Stimulus', 'stim'}}, 10, [], []);
        cfg.setName('stim');
        fit = Analysis.RunAnalysisForNeuron(trial, 1, ConfigColl({cfg}));
        summary = FitResSummary({fit});
        analysisAIC = fit.AIC(1);
        analysisBIC = fit.BIC(1);
        analysisLogLL = fit.logLL(1);
        analysisCoeffs = fit.getCoeffs(1)';
        analysisLambdaHead = fit.lambda.data(1:5, 1)';
        analysisSummaryAIC = summary.AIC(1);
        analysisSummaryBIC = summary.BIC(1);
        """,
        nargout=0,
    )
    return {
        "aic": _to_numpy(engine.workspace["analysisAIC"]).reshape(-1),
        "bic": _to_numpy(engine.workspace["analysisBIC"]).reshape(-1),
        "logll": _to_numpy(engine.workspace["analysisLogLL"]).reshape(-1),
        "coeffs": _to_numpy(engine.workspace["analysisCoeffs"]).reshape(-1),
        "lambda_head": _to_numpy(engine.workspace["analysisLambdaHead"]).reshape(-1),
        "summary_aic": _to_numpy(engine.workspace["analysisSummaryAIC"]).reshape(-1),
        "summary_bic": _to_numpy(engine.workspace["analysisSummaryBIC"]).reshape(-1),
    }


__all__ = [
    "matlab_engine_available",
    "run_analysis_reference",
    "run_point_process_reference",
    "run_simulated_network_reference",
    "start_matlab_engine",
]
