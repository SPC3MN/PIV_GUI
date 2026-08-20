"""piv_suite -- unified planar/stereo, CPU/GPU PIV processing core.

Consolidates the previously-duplicated pipelines from SPC3MN's
Stereo_PIV_GPU, Planar_PIV_GPU, Planar_PIV_CPU, and Stereo_PIV_CPU repos
into one package with pluggable engines (engines/), shared post-processing
(processing/), a canonical settings schema (config/), image/.set ingestion
(io/), stereo calibration (calibration/), plotting (plotting/), and a CLI
(cli/) that reproduces the four original scripts behind one entry point.
"""

__version__ = "0.2.1"
