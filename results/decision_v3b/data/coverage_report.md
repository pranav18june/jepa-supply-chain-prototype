| split | transitions | zero_request_frac | max_request_frac | truncated_request_frac | truncated_by_capacity_frac | capacity_saturated_frac | backlog_positive_frac | severe_backlog_frac | recovery_events | lastmile_wh_empty_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 14400 | 0.397 | 0.017 | 0.241 | 0.118 | 0.135 | 0.066 | 0.029 | 122 | 0.485 |
| validation | 2880 | 0.409 | 0.017 | 0.226 | 0.116 | 0.130 | 0.078 | 0.027 | 35 | 0.507 |
| test | 2880 | 0.393 | 0.017 | 0.257 | 0.099 | 0.114 | 0.122 | 0.049 | 33 | 0.522 |
| test_ood | 2880 | 0.076 | 0.007 | 0.459 | 0.018 | 0.024 | 0.436 | 0.318 | 35 | 0.919 |

| train behavior | transitions | zero_request_frac | max_request_frac | truncated_request_frac | truncated_by_capacity_frac | capacity_saturated_frac | backlog_positive_frac | severe_backlog_frac | recovery_events | lastmile_wh_empty_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| basestock | 2400 | 0.193 | 0.005 | 0.015 | 0.015 | 0.018 | 0.088 | 0.016 | 34 | 0.168 |
| intermittent | 2400 | 0.717 | 0.000 | 0.215 | 0.155 | 0.176 | 0.117 | 0.093 | 6 | 0.603 |
| mixed_item | 2400 | 0.586 | 0.018 | 0.308 | 0.090 | 0.117 | 0.028 | 0.006 | 16 | 0.846 |
| native_noise | 2400 | 0.293 | 0.002 | 0.011 | 0.010 | 0.016 | 0.010 | 0.000 | 4 | 0.159 |
| starve_recover | 2400 | 0.518 | 0.001 | 0.017 | 0.017 | 0.035 | 0.153 | 0.058 | 62 | 0.161 |
| uniform | 2400 | 0.078 | 0.075 | 0.880 | 0.422 | 0.445 | 0.000 | 0.000 | 0 | 0.975 |

Unseen-cell test mass: iid 0.0007, ood 0.2653

Checks: {'backlog_positive_frac': False, 'truncated_request_frac': True, 'capacity_saturated_frac': True, 'zero_request_frac': True, 'max_request_frac': False, 'recovery_events': True, 'max_behavior_share': True, 'test_mass_in_unseen_cells': True}
Coverage gate pass: False