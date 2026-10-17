| split | transitions | zero_request_frac | max_request_frac | truncated_request_frac | truncated_by_capacity_frac | capacity_saturated_frac | backlog_positive_frac | severe_backlog_frac | recovery_events | lastmile_wh_empty_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 24000 | 0.431 | 0.029 | 0.253 | 0.127 | 0.140 | 0.136 | 0.076 | 201 | 0.490 |
| validation | 3600 | 0.448 | 0.033 | 0.256 | 0.131 | 0.146 | 0.157 | 0.094 | 24 | 0.480 |
| test | 3600 | 0.450 | 0.027 | 0.231 | 0.127 | 0.143 | 0.147 | 0.093 | 34 | 0.470 |
| test_ood | 2880 | 0.073 | 0.011 | 0.492 | 0.026 | 0.030 | 0.385 | 0.261 | 31 | 0.945 |

| train behavior | transitions | zero_request_frac | max_request_frac | truncated_request_frac | truncated_by_capacity_frac | capacity_saturated_frac | backlog_positive_frac | severe_backlog_frac | recovery_events | lastmile_wh_empty_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| basestock | 4020 | 0.167 | 0.005 | 0.012 | 0.012 | 0.018 | 0.196 | 0.026 | 58 | 0.159 |
| intermittent | 4020 | 0.699 | 0.000 | 0.212 | 0.175 | 0.197 | 0.114 | 0.090 | 12 | 0.642 |
| mixed_item | 3960 | 0.589 | 0.035 | 0.356 | 0.105 | 0.126 | 0.012 | 0.002 | 15 | 0.848 |
| native_noise | 4020 | 0.315 | 0.002 | 0.013 | 0.013 | 0.019 | 0.005 | 0.000 | 8 | 0.185 |
| starve_recover | 3960 | 0.702 | 0.009 | 0.045 | 0.043 | 0.050 | 0.489 | 0.342 | 108 | 0.132 |
| uniform | 4020 | 0.123 | 0.125 | 0.880 | 0.413 | 0.428 | 0.000 | 0.000 | 0 | 0.977 |

Unseen-cell test mass: iid 0.0000, ood 0.0045

Checks: {'backlog_positive_frac': True, 'truncated_request_frac': True, 'capacity_saturated_frac': True, 'zero_request_frac': True, 'max_request_frac': True, 'recovery_events': True, 'max_behavior_share': True, 'test_mass_in_unseen_cells': True}
Coverage gate pass: True