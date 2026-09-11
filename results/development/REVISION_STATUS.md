# Latest revision status

The three-channel generative development study has run. V6 maximum fusion measured AUROC 0.6291; V7 mean fusion reached 0.7181 on the same six-family benchmark. V7 TPR is 45.14%, FPR 12.5% (two of sixteen clean images), and F1 0.6190. This is development-selected evidence, not final validation.

DINO remains blocked by the numerical runtime. No security restriction was disabled. All eight terrain classes remain normal, and no other dataset was downloaded. Final holdout remains untouched.

Detailed status: `revision_status_latest.json`. Results: `generative_v6/`, `generative_v7/`, `rare_prevalence/`. Submission evidence: `../../report/TCMD_SS_Report.pdf` and the root/notebooks notebooks. Fourteen isolated tests pass; full numerical test collection remains blocked.
