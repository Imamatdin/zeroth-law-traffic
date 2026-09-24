# Initial issue backlog

Create these as GitHub Issues once the private remote is available. Each item should close with a visible artifact or check, not a research-only update.

| Priority | Issue | Acceptance evidence |
| --- | --- | --- |
| P0 | Organizer harness runs | Official starter kit committed unchanged; documented command starts |
| P0 | Valid zero-score baseline | Official validator accepts output for every provided sample |
| P0 | Annotate all samples | Timestamped event labels with reviewer notes |
| P0 | Map camera geometry | Versioned lane, stop-line, and signal map for each camera |
| P0 | Detector baseline | Representative-frame misses, bad boxes, and runtime logged |
| P0 | Tracker baseline | Visual ID continuity check and runtime logged |
| P0 | Submission verification | Repeatable schema, bounds, and file-presence checks |
| P1 | Wrong-way detection | Scored against reviewed labels and boundary examples |
| P1 | Stopped-vehicle detection | Scored with queue false-positive review |
| P1 | Jaywalking and line crossing | Geometry rules plus scored examples |
| P1 | Red-light and stop-line events | Signal visibility verified; scored or explicitly marked infeasible |
| P1 | Congestion detection | Operational definition and scored examples |
| P1 | Causal risk baseline | Time-aligned risk output and evaluator result |
| P1 | Website result viewer | Real pipeline output drives video, timeline, and risk chart |

Defer P2 accident/near-miss verification, TensorRT, and learned temporal risk until the P0 pipeline is valid and P1 scoring evidence exists.
