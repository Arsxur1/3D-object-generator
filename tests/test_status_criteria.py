"""ACNS electrographic status epilepticus criteria (layer4_detect/status_criteria.py)."""

from neurolens.layer4_detect.status_criteria import max_seizure_time_in_window, status_epilepticus


def test_short_seizures_are_not_status():
    # the old 30-s rule declared status after half a minute of seizure activity
    assert not status_epilepticus([(10, 45)])["suspected"]
    assert not status_epilepticus([(0, 134), (900, 1000)])["suspected"]


def test_continuous_ten_minutes_adult():
    r = status_epilepticus([(100, 100 + 600)])
    assert r["suspected"] and r["criterion"] == "continuous_10min"
    assert not status_epilepticus([(100, 100 + 600)], neonate=True)["suspected"]  # neonatal rule is hourly only


def test_hourly_burden_adult_and_neonate():
    # 13 seizures of 60 s within one hour = 13 min >= 20% of 60 min
    iv = [(i * 270.0, i * 270.0 + 60.0) for i in range(13)]
    assert max_seizure_time_in_window(iv) == 780.0
    r = status_epilepticus(iv)
    assert r["suspected"] and r["criterion"] == "hourly_burden"
    assert not status_epilepticus(iv, neonate=True)["suspected"]  # neonates need >= 50% (30 min)
    dense = [(i * 100.0, i * 100.0 + 60.0) for i in range(36)]  # 36 min of seizures in 1 h
    assert status_epilepticus(dense, neonate=True)["suspected"]


def test_same_seizure_time_spread_over_hours_is_not_status():
    iv = [(h * 3600.0 + 10, h * 3600.0 + 400) for h in range(6)]  # 6.5 min per hour
    assert not status_epilepticus(iv)["suspected"]
