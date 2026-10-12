# FLEX runtime files

These files are copied without modification from the FLEX implementation used
by the main active-dynamics-SNN checkout. The upstream Git revision was not
retained in that local copy. The hashes below pin the exact runtime used here.

Only the three runtime modules needed by `actdyn.policy.baseline_flex` are
included, with the original MIT license. This makes them available to
`git archive` and avoids a dependency on an ignored local directory.
No numerical code is changed.

- `agent.py`: `8a6ccb3c238c1430674b8a286806d8449af2a1fab074611680e82db3a20d4fb1`
- `computations.py`: `9093752e83a50283797d4bc8b9af907a7915eeb71af185f7cc0e86dce8dfdc29`
- `policies.py`: `fa04b5266d54b3f693de8011e288fdc6d52ecc24d699f06d3cfcafeb3941e9fe`
- `LICENSE`: `7dbee558af3e7347fbea9af61d05973bbde09421bb887c69209aa31897bdf343`
