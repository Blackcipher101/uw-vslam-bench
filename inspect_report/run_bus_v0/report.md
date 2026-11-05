# Underwater inspection report (draft)

> **AI-GENERATED DRAFT - detections come from an open-vocabulary detector and text from a vision-language model. Severity values are suggestions only and MUST be reviewed by a qualified human inspector.**

## Summary

Automatic draft covering 585 s, 198 keyframes, track length 65 m. The detector proposed 30 candidate findings, but none was confirmed by the visual verification step, so no finding is reported as confirmed. The candidates are listed below as unconfirmed for human review.

## Coverage

- Source: `bus_outside`
- Duration: 585 s, keyframes: 198 (198 with pose)
- Track length: 64.5 m (VIO trajectory: `/home/nevle.nehal/uw-vslam-bench/results/bus_outside/uwct/s4_rep.tum`)
- Detector: google/owlv2-base-patch16-ensemble; VLM: Qwen/Qwen3-VL-4B-Instruct; score threshold 0.15 (base prompts)

## Findings

Positions are in the VIO world frame (metres, start-camera frame, not gravity aligned). Sizes are approximate box extents from stereo depth. Frame refs are keyframe ids.

| ID | Type | Status | Location (x,y,z m) | Size | Conf. | Obs. | Frames | Severity (AI suggestion) | Assessment |
|---|---|---|---|---|---|---|---|---|---|
| F01 | tire | unconfirmed (rejected by VLM check) | (4.2, 0.5, 0.1) | 63 x 95 cm | 0.72 | 10 | kf_0080, kf_0081, kf_0082... | n/a (not confirmed) | The object appears to be a dark, irregularly shaped mass with no clear tire features. |
| F02 | fish | unconfirmed (rejected by VLM check) | (5.7, 9.1, 1.0) | 28 x 21 cm | 0.62 | 5 | kf_0040, kf_0041, kf_0043... | n/a (not confirmed) | The shape is indistinct and lacks clear features to confirm it is a fish. |
| F03 | fish | unconfirmed (rejected by VLM check) | (4.0, 10.3, -0.4) | 15 x 17 cm | 0.48 | 27 | kf_0030, kf_0030, kf_0030... | n/a (not confirmed) | The object is too blurry and indistinct to be confidently identified as a fish. |
| F04 | tire | unconfirmed (rejected by VLM check) | (3.3, 7.2, 0.1) | 86 x 127 cm | 0.44 | 1 | kf_0026 | n/a (not confirmed) | The object is indistinct and appears to be submerged in murky water. |
| F05 | marine_growth | unconfirmed (rejected by VLM check) | (3.6, -1.2, 0.3) | 63 x 129 cm | 0.44 | 20 | kf_0088, kf_0089, kf_0090... | n/a (not confirmed) | The surface appears covered with some form of biological growth, but it is not clearly identifiable as marine growth. |
| F06 | window | unconfirmed (rejected by VLM check) | (2.0, -0.2, 0.3) | 106 x 165 cm | 0.43 | 12 | kf_0000, kf_0001, kf_0002... | n/a (not confirmed) | The structure is heavily obscured by greenish water and sediment, making features indistinct. |
| F07 | marine_growth | unconfirmed (rejected by VLM check) | (3.1, -2.6, -0.5) | 217 x 86 cm | 0.42 | 9 | kf_0093, kf_0094, kf_0095... | n/a (not confirmed) | The image is too blurry and indistinct to confirm marine growth. |
| F08 | marine_growth | unconfirmed (rejected by VLM check) | (3.2, 9.8, -0.4) | 84 x 48 cm | 0.41 | 15 | kf_0030, kf_0031, kf_0031... | n/a (not confirmed) | The image shows no clear signs of marine growth. |
| F09 | fish | unconfirmed (rejected by VLM check) | (2.0, -1.3, 0.1) | 16 x 24 cm | 0.41 | 2 | kf_0109, kf_0183 | n/a (not confirmed) | The image is too blurry to identify any specific object or creature. |
| F10 | marine_growth | unconfirmed (rejected by VLM check) | (4.7, -1.4, -0.4) | 209 x 85 cm | 0.40 | 16 | kf_0086, kf_0087, kf_0088... | n/a (not confirmed) | The image is too blurry to confirm marine growth presence. |
| F11 | marine_growth | unconfirmed (rejected by VLM check) | (2.2, -1.2, 0.2) | 58 x 149 cm | 0.39 | 3 | kf_0180, kf_0182, kf_0183 | n/a (not confirmed) | The image shows a shipwreck covered in marine growth, but the growth is not clearly identifiable or extensive. |
| F12 | window | unconfirmed (rejected by VLM check) | (4.8, 5.4, 1.0) | 279 x 334 cm | 0.38 | 9 | kf_0140, kf_0141, kf_0142... | n/a (not confirmed) | The image shows a murky underwater environment with indistinct shapes that may resemble windows but are not clearly identifiable as such. |
| F13 | fish | unconfirmed (rejected by VLM check) | n/a (no pose or depth) | n/a | 0.38 | 1 | kf_0029 | n/a (not confirmed) | The image is too blurry to identify any clear biological features. |
| F14 | marine_growth | unconfirmed (rejected by VLM check) | (2.3, -2.9, -0.5) | 90 x 35 cm | 0.38 | 7 | kf_0180, kf_0181, kf_0181... | n/a (not confirmed) | The image is too blurry to determine if marine growth is present. |
| F15 | window | unconfirmed (rejected by VLM check) | (4.9, 4.6, 0.8) | 170 x 236 cm | 0.37 | 3 | kf_0067, kf_0067, kf_0068 | n/a (not confirmed) | The structure appears to be a window frame with significant surface degradation and obscured visibility. |
| F16 | marine_growth | unconfirmed (rejected by VLM check) | (6.7, 4.9, -0.6) | 247 x 87 cm | 0.36 | 2 | kf_0156, kf_0157 | n/a (not confirmed) | The image is too blurry and murky to determine if marine growth is present. |
| F17 | crack | unconfirmed (rejected by VLM check) | n/a (no pose or depth) | n/a | 0.36 | 2 | kf_0169, kf_0170 | n/a (not confirmed) | The image is too blurry to confirm if the line is a crack. |
| F18 | marine_growth | unconfirmed (rejected by VLM check) | (4.6, 3.2, 0.3) | 117 x 138 cm | 0.36 | 5 | kf_0072, kf_0073, kf_0075... | n/a (not confirmed) | The image shows a submerged structure with greenish water and some surface texture, but no clear marine growth is visible. |
| F19 | marine_growth | unconfirmed (rejected by VLM check) | (5.7, 3.6, -0.4) | 119 x 71 cm | 0.35 | 6 | kf_0158, kf_0159, kf_0160... | n/a (not confirmed) | The scene appears to be a typical underwater seabed with no obvious signs of marine growth. |
| F20 | crack | unconfirmed (rejected by VLM check) | (2.0, -1.6, -0.1) | 76 x 29 cm | 0.35 | 16 | kf_0099, kf_0102, kf_0103... | n/a (not confirmed) | The line appears subtle and not clearly defined as a crack. |
| F21 | debris | unconfirmed (rejected by VLM check) | n/a (no pose or depth) | n/a | 0.35 | 1 | kf_0029 | n/a (not confirmed) | The object appears indistinct and poorly defined due to low image resolution. |
| F22 | marine_growth | unconfirmed (rejected by VLM check) | (5.1, -0.6, -0.5) | 215 x 96 cm | 0.34 | 4 | kf_0168, kf_0169, kf_0169... | n/a (not confirmed) | The image is too blurry and murky to determine if marine growth is present. |
| F23 | fish | unconfirmed (rejected by VLM check) | (6.4, 1.8, -0.5) | 30 x 19 cm | 0.34 | 3 | kf_0158, kf_0159, kf_0160 | n/a (not confirmed) | The shape is indistinct and cannot be confidently identified as a fish. |
| F24 | window | unconfirmed (rejected by VLM check) | n/a (no pose or depth) | n/a | 0.34 | 1 | kf_0119 | n/a (not confirmed) | The image is heavily obscured by water turbidity, making it impossible to confirm if any structure is a window. |
| F25 | hole_damage | unconfirmed (rejected by VLM check) | (5.6, 9.0, 0.8) | 9 x 13 cm | 0.34 | 6 | kf_0042, kf_0043, kf_0044... | n/a (not confirmed) | The image is too blurry to determine if it represents a hole or damage. |
| F26 | crack | unconfirmed (rejected by VLM check) | n/a (no pose or depth) | n/a | 0.34 | 2 | kf_0172, kf_0173 | n/a (not confirmed) | The line appears to be a minor imperfection or shadow, not a clear crack. |
| F27 | debris | unconfirmed (rejected by VLM check) | n/a (no pose or depth) | n/a | 0.34 | 1 | kf_0059 | n/a (not confirmed) | The image is indistinct and lacks clear features to identify debris. |
| F28 | fish | unconfirmed (rejected by VLM check) | (2.8, 5.7, -0.2) | 39 x 27 cm | 0.34 | 17 | kf_0014, kf_0015, kf_0016... | n/a (not confirmed) | The image is too blurry and low-resolution to confirm if it is a fish. |
| F29 | hole_damage | unconfirmed (rejected by VLM check) | (2.0, -1.3, 0.6) | 19 x 32 cm | 0.34 | 13 | kf_0102, kf_0103, kf_0104... | n/a (not confirmed) | The image shows a dark circular area that may be a void or shadow, but no clear evidence of structural damage is visible. |
| F30 | marine_growth | unconfirmed (rejected by VLM check) | n/a (no pose or depth) | n/a | 0.33 | 1 | kf_0029 | n/a (not confirmed) | The image is too blurry to determine if it is marine growth or another object. |

Crops: `crops/<ID>.jpg` in the output directory (embedded in the HTML version).

## Faithfulness check

- Report-text sentences checked: 3; hallucinated statements: **0** (unknown ids 0, absent-class mentions 0, id/class mismatches 0, cites of unconfirmed findings 0)
- Findings not mentioned in the text: none
