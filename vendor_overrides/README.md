# External repository overrides

This directory contains only the project-specific files used on top of the official TinyFormer and DEIMv2 repositories. It is deliberately not a full vendor copy.

To reconstruct an external backend, clone the upstream repository at the commit recorded in `reports/environments/`, then copy the corresponding override directory over the clone while preserving relative paths.

The modified evaluator files add the common COCO prediction export required for comparable A/B/COMBINED metrics. The solver modifications add benchmark-compatible checkpoint and early-stopping behaviour. The configuration files define the one-class Ganoderma datasets, disabled native augmentation policy, and B0/B1/B2 run variants.

The original upstream license is included inside each override directory for attribution and redistribution terms.
