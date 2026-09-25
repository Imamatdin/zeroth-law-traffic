# Tests

Run `python -B -m unittest discover -s tests -v` after installing `requirements-dev.txt`.

Current checks cover internal shapes/ranges, official-label consistency, cache schema/provenance/reuse, saved-prediction evaluation and error paths, and organizer-script byte identity. Evaluation tests use the unchanged organizer examples; cache tests are storage unit tests. They do not measure detection or anticipation quality and do not generate video.

Add segment-engine tests and real implementation determinism/causality tests when those components exist. Repeated scoring of an example fixture is not a model determinism test.
