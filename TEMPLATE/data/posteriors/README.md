# Posterior store

Fitted-model outputs from the calibration workbench land here as ArviZ
InferenceData NetCDF files (`*.nc`), e.g. `measurement_v1.nc`.

The runtime Assessor loads draws from these files instead of running MCMC.
`.nc` files are git-ignored because they can be large and are fully
regenerable from `calibration/` scripts + raw data.
