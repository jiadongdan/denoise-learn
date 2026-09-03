"""Provider protocol constants shared by the CLI and worker."""

PROVIDER_CONTRACT_VERSION = "denoise-learn-provider-v1"
WORKER_SCHEMA_VERSION = "scientific-denoise-worker-v1"

COMPARISON_POLICIES = {
    "raw_unit_space",
    "clipped_0_1",
    "minmax_0_1",
}
