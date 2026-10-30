# Exact decoded-RGB duplicate audit

All ETH-80 and Freiburg source images and the original seed42 experimental partitions (Freiburg author split0).

| Dataset | Images decoded | Unique RGB arrays | Duplicate groups | Cross-split duplicate pairs | Conflicting-label groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| eth80 | 3280 | 3280 | 0 | 0 | 0 |
| freiburg | 4947 | 4933 | 14 | 9 | 0 |

Exact SHA-256 equality of decoded RGB pixel arrays including mode and image dimensions; distinct source IDs only.

This audit does not test perceptual near-duplicates, common physical product instances, or other Freiburg fold memberships.

Source archives and original experimental manifests were checksum-verified. No samples, splits, models or classifier results were changed or used. Full IDs and split memberships for any duplicate groups appear in exact-pixel-duplicate-audit.json.
