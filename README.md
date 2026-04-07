# heart-sound-segmentation
Noise-robust PCG heart sound segmentation using Average Shannon Energy and Z-score noise detection
# Heart Sound Segmentation
Automatic S1/S2 segmentation of phonocardiogram (PCG) recordings using
Average Shannon Energy and statistical noise rejection.

## Overview
This pipeline takes a raw heart sound `.wav` file and:
- Filters it to the 40–500 Hz cardiac frequency range
- Computes a Normalized Average Shannon Energy (NASE) envelope
- Removes noisy segments using z-score outlier detection (cutoff = 2.75)
- Validates remaining sound lobes against physiological constraints
- Labels each lobe as S1 or S2 using bidirectional correlation propagation

## Results (heart_sound.wav)
| Metric | This Implementation | Paper Benchmark |
|---|---|---|
| S1 Sensitivity | 92.31% | 97.22% |
| S2 Sensitivity | 100.00% | 97.22% |
| Overall Sensitivity | 96.15% | 97.44% |
| Segmentation | SUCCESS | — |

## Requirements
Python 3.8 or above

Install dependencies:
```
pip install numpy librosa scipy matplotlib
```
## Usage
Run the scripts in order:
```python
python preprocess_01.py
python shannon_energy_02.py
python noisy_lobe_03.py
python lobe_validation_04.py
python s1_s2_identify_05.py
python visualization_evaluation_06.py
```
Set `FILE_PATH = "your_file.wav"` in each script before running.

## Citation
If you use this code, please cite:

> [Divya Keerthi] (2025). Heart Sound Segmentation using Average Shannon Energy.
> GitHub. https://github.com/divyakeerthi-44/heart-sound-segmentation

This implementation is based on the method described in:

> S. Springer et al., "Logistic Regression-HSMM-based Heart Sound
> Segmentation," IEEE Transactions on Biomedical Engineering,
> vol. 63, no. 4, pp. 822–832, 2016.

## License
MIT License — free to use and modify with attribution.
