"""TrialConfig demo aligned to MATLAB helpfiles/TrialConfigExamples.m."""

from __future__ import annotations

from nstat.compat.matlab import ConfigColl, TrialConfig


def run_demo() -> ConfigColl:
    # MATLAB reference:
    # tc1 = TrialConfig({'Force','f_x'},2000,[.1 .2],-1,2);
    # tc2 = TrialConfig({'Position','x'},2000,[.1 .2],-1,2);
    tc1 = TrialConfig(["Force", "f_x"], 2000.0, [0.1, 0.2], -1.0, 2.0)
    tc2 = TrialConfig(["Position", "x"], 2000.0, [0.1, 0.2], -1.0, 2.0)
    tcc = ConfigColl([tc1, tc2])
    return tcc


if __name__ == "__main__":
    collection = run_demo()
    print("Config names:", collection.getConfigNames())
