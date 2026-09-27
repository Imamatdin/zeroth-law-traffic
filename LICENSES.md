# Licences and attribution

Primary sources checked 2026-09-27. Project original-code licence is **UNDECIDED** pending team confirmation. This inventory does not relicense contributors' work or organizer starter files. Dataset rights are separate.

| Component | Licence | Primary source / qualification |
|---|---|---|
| Ultralytics code/headless distribution and YOLO11m weights | AGPL-3.0 open-source option | [Model card](https://huggingface.co/Ultralytics/YOLO11), [licensing](https://www.ultralytics.com/license), [code licence](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). Copy: third_party/Ultralytics-AGPL-3.0.txt. No enterprise licence claimed |
| Original ByteTrack | MIT, Yifu Zhang 2021 | [Licence](https://github.com/FoundationVision/ByteTrack/blob/main/LICENSE), copy: third_party/ByteTrack-MIT.txt. We import the **Ultralytics implementation under its AGPL distribution**, not a separately installed MIT implementation |
| PyTorch / torchvision | BSD-style / BSD-3-Clause plus bundled third-party notices | [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [vision](https://github.com/pytorch/vision/blob/main/LICENSE) |
| OpenCV | Apache-2.0 for modern OpenCV, plus wheel third-party notices | [Source licence](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| NumPy / pandas | BSD-3-Clause | [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt), [pandas](https://github.com/pandas-dev/pandas/blob/main/LICENSE) |
| PyYAML | MIT | [Licence](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| lap | BSD-2-Clause | [Licence](https://github.com/gatagat/lap/blob/master/LICENSE) |
| imageio-ffmpeg wrapper | BSD-2-Clause | [Licence](https://github.com/imageio/imageio-ffmpeg/blob/main/LICENSE) |
| Bundled FFmpeg binary | Build-dependent LGPL/GPL, not wrapper BSD | [FFmpeg legal](https://ffmpeg.org/legal.html). Inspect installed binary with -L and -buildconf; GPL-enabled builds retain GPL/codecs obligations |
| NVIDIA CUDA/cuDNN wheel libraries | Component-specific NVIDIA terms | [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html); preserve wheel licence/notices. PyTorch's BSD does not relicense CUDA |
| COCO | Annotation CC BY 4.0; images individually owned | [Terms](https://cocodataset.org/#termsofuse), DATASETS.md |

The [version-specific publisher metadata inventory](docs/dependency_licenses.md) records all 61 locked dependencies. The hash lock also includes transitive dependencies. Retain their installed .dist-info licence/notice files when distributing an environment. The weights release contains only the checkpoint, checksum and model licence, not a vendored Python/CUDA environment.

Provide corresponding source for AGPL-covered components/modifications, including applicable network-use obligations; preserve upstream notices. Original project licensing still requires team confirmation: public visibility alone is not a licence grant. The organizer starter kit has no separately supplied licence; contest use is described in docs/task_spec.md. Do not label organizer footage or all COCO images freely redistributable.
