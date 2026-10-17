| split | transitions | zero_request_frac | max_request_frac | truncated_request_frac | truncated_by_capacity_frac | capacity_saturated_frac | backlog_positive_frac | severe_backlog_frac | recovery_events | lastmile_wh_empty_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 14400 | 0.365 | 0.002 | 0.300 | 0.163 | 0.179 | 0.050 | 0.020 | 63 | 0.532 |
| validation | 2880 | 0.369 | 0.003 | 0.308 | 0.149 | 0.163 | 0.077 | 0.013 | 16 | 0.515 |
| test | 2880 | 0.345 | 0.003 | 0.290 | 0.143 | 0.154 | 0.044 | 0.001 | 12 | 0.555 |
| test_ood | 2880 | 0.072 | 0.015 | 0.525 | 0.027 | 0.031 | 0.473 | 0.346 | 29 | 0.940 |

| train behavior | transitions | zero_request_frac | max_request_frac | truncated_request_frac | truncated_by_capacity_frac | capacity_saturated_frac | backlog_positive_frac | severe_backlog_frac | recovery_events | lastmile_wh_empty_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| basestock | 2880 | 0.203 | 0.005 | 0.013 | 0.013 | 0.018 | 0.113 | 0.011 | 42 | 0.146 |
| intermittent | 2880 | 0.707 | 0.000 | 0.224 | 0.172 | 0.198 | 0.120 | 0.089 | 12 | 0.577 |
| mixed_item | 2880 | 0.585 | 0.002 | 0.337 | 0.132 | 0.159 | 0.017 | 0.002 | 9 | 0.855 |
| native_noise | 2880 | 0.329 | 0.001 | 0.010 | 0.010 | 0.018 | 0.000 | 0.000 | 0 | 0.110 |
| uniform | 2880 | 0.001 | 0.000 | 0.916 | 0.487 | 0.502 | 0.000 | 0.000 | 0 | 0.976 |

Unseen-cell test mass: iid 0.0000, ood 0.3226

Checks: {'backlog_positive_frac': False, 'truncated_request_frac': True, 'capacity_saturated_frac': True, 'zero_request_frac': True, 'max_request_frac': False, 'recovery_events': False, 'max_behavior_share': True, 'test_mass_in_unseen_cells': True}
Coverage gate pass: False