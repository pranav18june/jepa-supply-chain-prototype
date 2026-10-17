| split | transitions | zero_request_frac | max_request_frac | truncated_request_frac | truncated_by_capacity_frac | capacity_saturated_frac | backlog_positive_frac | severe_backlog_frac | recovery_events | lastmile_wh_empty_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 14400 | 0.450 | 0.030 | 0.244 | 0.112 | 0.126 | 0.114 | 0.071 | 96 | 0.485 |
| validation | 2880 | 0.451 | 0.027 | 0.233 | 0.123 | 0.137 | 0.105 | 0.074 | 15 | 0.458 |
| test | 2880 | 0.428 | 0.027 | 0.248 | 0.159 | 0.173 | 0.144 | 0.076 | 19 | 0.528 |
| test_ood | 2880 | 0.072 | 0.017 | 0.487 | 0.024 | 0.032 | 0.459 | 0.350 | 26 | 0.926 |

| train behavior | transitions | zero_request_frac | max_request_frac | truncated_request_frac | truncated_by_capacity_frac | capacity_saturated_frac | backlog_positive_frac | severe_backlog_frac | recovery_events | lastmile_wh_empty_frac |
|---|---|---|---|---|---|---|---|---|---|---|
| basestock | 2400 | 0.206 | 0.007 | 0.014 | 0.014 | 0.018 | 0.091 | 0.006 | 19 | 0.142 |
| intermittent | 2400 | 0.701 | 0.000 | 0.197 | 0.154 | 0.184 | 0.070 | 0.052 | 6 | 0.648 |
| mixed_item | 2400 | 0.608 | 0.033 | 0.336 | 0.098 | 0.119 | 0.008 | 0.003 | 4 | 0.853 |
| native_noise | 2400 | 0.340 | 0.002 | 0.010 | 0.010 | 0.018 | 0.006 | 0.000 | 4 | 0.150 |
| starve_recover | 2400 | 0.722 | 0.009 | 0.037 | 0.036 | 0.043 | 0.509 | 0.367 | 63 | 0.140 |
| uniform | 2400 | 0.123 | 0.126 | 0.871 | 0.357 | 0.372 | 0.000 | 0.000 | 0 | 0.977 |

Unseen-cell test mass: iid 0.0417, ood 0.1833

Checks: {'backlog_positive_frac': True, 'truncated_request_frac': True, 'capacity_saturated_frac': True, 'zero_request_frac': True, 'max_request_frac': True, 'recovery_events': False, 'max_behavior_share': True, 'test_mass_in_unseen_cells': True}
Coverage gate pass: False