from pathlib import Path

import numpy as np

FX = Path(__file__).resolve().parent / "fixtures"
EPISODES = ["e2e_run/episode_pre_place_1_osc_0.npz", "demo/episode_gv_07_0.npz"]
FILES = [
    "e2e_run/index.json",
    "e2e_run/recordings/episode_pre_place_1_osc_0/meta.json",
    "e2e_run/recordings/episode_pre_place_1_osc_0/chunks/chunk_00000.npz",
    "e2e_suite/hard_suite.json",
    "e2e_suite/hard_suite.md",
    "e2e_suite/summarize.out",
    "e2e_suite/suite.json",
    "miles/buffer.json",
    "miles/miles_report.md",
    "train/train.log",
    "train/refit_report.json",
    "train/config.yaml",
    "buffer_W150all_head.json",
    "grasp_set2_index.json",
    "pre_place_1.npz",
    "arms.json",
    "KEYS.md",
    *EPISODES,
]


def test_files_exist_and_npz_load():
    for f in FILES:
        assert (FX / f).is_file(), f
    assert len(list((FX / "miles/miles").glob("*.npz"))) == 3
    for p in FX.rglob("*.npz"):
        assert np.load(p, allow_pickle=True).files, p


def test_episode_T_and_cloud():
    for f in EPISODES:
        z = np.load(FX / f, allow_pickle=True)
        assert len(z["utime"]) == 40 and len(z["sim_time"]) == 40, f
        assert z["pcd_belt"].shape[1] <= 128
        assert all(len(x) <= 128 for x in z["pcd"])


def test_total_size():
    assert sum(p.stat().st_size for p in FX.rglob("*") if p.is_file()) < 5_000_000


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
    print("ok")
