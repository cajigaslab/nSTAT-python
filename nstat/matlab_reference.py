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
        netSpikeCounts = [sum(yout(:,1)>.5), sum(yout(:,2)>.5)];
        netProbHead = yout(1:5,3:4);
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


__all__ = [
    "matlab_engine_available",
    "run_point_process_reference",
    "run_simulated_network_reference",
    "start_matlab_engine",
]
