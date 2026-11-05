# Underwater inspection report (draft)

> **AI-GENERATED DRAFT - detections come from an open-vocabulary detector and text from a vision-language model. Severity values are suggestions only and MUST be reviewed by a qualified human inspector.**

## Summary

This automatic draft covers 585 seconds of inspection across 198 keyframes and 65 meters of track, identifying 30 structured findings. Key findings include [F01] (tire), [F02] (fish), [F03] (fish), [F05] (marine_growth), [F06] (window), [F07] (marine_growth), [F08] (marine_growth), [F11] (marine_growth), [F15] (window), [F16] (marine_growth), [F17] (crack), [F20] (crack), [F25] (hole_damage), [F29] (hole_damage), and others. Many findings are low-confidence or indistinct due to murky water, blur, or poor resolution. This result requires human review to validate object identity, assess reliability, and determine actionable insights.

## Coverage

- Source: `AFRL bus_outside (sunken bus, Fantasy Lake NC)`
- Duration: 585 s, keyframes: 198 (198 with pose)
- Track length: 64.5 m (VIO trajectory: `/home/griffinlabs/uwvio/data/bus_outside_traj_s4_rep.tum`)
- Detector: google/owlv2-base-patch16-ensemble; VLM: Qwen/Qwen3-VL-4B-Instruct; score threshold 0.15 (base prompts)

## Findings

Positions are in the VIO world frame (metres, start-camera frame, not gravity aligned). Sizes are approximate box extents from stereo depth. Frame refs are keyframe ids.

| ID | Type | Location (x,y,z m) | Size | Conf. | Obs. | Frames | Severity (AI suggestion) | Assessment |
|---|---|---|---|---|---|---|---|---|
| F01 | tire | (4.2, 0.5, 0.1) | 63 x 95 cm | 0.72 | 10 | kf_0080, kf_0081, kf_0082... | info | The object appears to be a dark, irregularly shaped mass with no clear tire features. |
| F02 | fish | (5.7, 9.1, 1.0) | 28 x 21 cm | 0.62 | 5 | kf_0040, kf_0041, kf_0043... | info | The shape is indistinct and lacks clear features to confirm it is a fish. |
| F03 | fish | (4.0, 10.3, -0.4) | 15 x 17 cm | 0.48 | 27 | kf_0030, kf_0030, kf_0030... | info | The object is too blurry and indistinct to be confidently identified as a fish. |
| F04 | tire | (3.3, 7.2, 0.1) | 86 x 127 cm | 0.44 | 1 | kf_0026 | info | The object is indistinct and appears to be submerged in murky water. |
| F05 | marine_growth | (3.6, -1.2, 0.3) | 63 x 129 cm | 0.44 | 20 | kf_0088, kf_0089, kf_0090... | low | The surface appears covered with some form of biological growth, but it is not clearly identifiable as marine growth. |
| F06 | window | (2.0, -0.2, 0.3) | 106 x 165 cm | 0.43 | 12 | kf_0000, kf_0001, kf_0002... | info | The structure is heavily obscured by greenish water and sediment, making features indistinct. |
| F07 | marine_growth | (3.1, -2.6, -0.5) | 217 x 86 cm | 0.42 | 9 | kf_0093, kf_0094, kf_0095... | none | The image is too blurry and indistinct to confirm marine growth. |
| F08 | marine_growth | (3.2, 9.8, -0.4) | 84 x 48 cm | 0.41 | 15 | kf_0030, kf_0031, kf_0031... | none | The image shows no clear signs of marine growth. |
| F09 | fish | (2.0, -1.3, 0.1) | 16 x 24 cm | 0.41 | 2 | kf_0109, kf_0183 | info | The image is too blurry to identify any specific object or creature. |
| F10 | marine_growth | (4.7, -1.4, -0.4) | 209 x 85 cm | 0.40 | 16 | kf_0086, kf_0087, kf_0088... | none | The image is too blurry to confirm marine growth presence. |
| F11 | marine_growth | (2.2, -1.2, 0.2) | 58 x 149 cm | 0.39 | 3 | kf_0180, kf_0182, kf_0183 | none | The image shows a shipwreck covered in marine growth, but the growth is not clearly identifiable or extensive. |
| F12 | window | (4.8, 5.4, 1.0) | 279 x 334 cm | 0.38 | 9 | kf_0140, kf_0141, kf_0142... | info | The image shows a murky underwater environment with indistinct shapes that may resemble windows but are not clearly identifiable as such. |
| F13 | fish | n/a (no pose or depth) | n/a | 0.38 | 1 | kf_0029 | info | The image is too blurry to identify any clear biological features. |
| F14 | marine_growth | (2.3, -2.9, -0.5) | 90 x 35 cm | 0.38 | 7 | kf_0180, kf_0181, kf_0181... | none | The image is too blurry to determine if marine growth is present. |
| F15 | window | (4.9, 4.6, 0.8) | 170 x 236 cm | 0.37 | 3 | kf_0067, kf_0067, kf_0068 | info | The structure appears to be a window frame with significant surface degradation and obscured visibility. |
| F16 | marine_growth | (6.7, 4.9, -0.6) | 247 x 87 cm | 0.36 | 2 | kf_0156, kf_0157 | none | The image is too blurry and murky to determine if marine growth is present. |
| F17 | crack | n/a (no pose or depth) | n/a | 0.36 | 2 | kf_0169, kf_0170 | none | The image is too blurry to confirm if the line is a crack. |
| F18 | marine_growth | (4.6, 3.2, 0.3) | 117 x 138 cm | 0.36 | 5 | kf_0072, kf_0073, kf_0075... | none | The image shows a submerged structure with greenish water and some surface texture, but no clear marine growth is visible. |
| F19 | marine_growth | (5.7, 3.6, -0.4) | 119 x 71 cm | 0.35 | 6 | kf_0158, kf_0159, kf_0160... | none | The scene appears to be a typical underwater seabed with no obvious signs of marine growth. |
| F20 | crack | (2.0, -1.6, -0.1) | 76 x 29 cm | 0.35 | 16 | kf_0099, kf_0102, kf_0103... | none | The line appears subtle and not clearly defined as a crack. |
| F21 | debris | n/a (no pose or depth) | n/a | 0.35 | 1 | kf_0029 | info | The object appears indistinct and poorly defined due to low image resolution. |
| F22 | marine_growth | (5.1, -0.6, -0.5) | 215 x 96 cm | 0.34 | 4 | kf_0168, kf_0169, kf_0169... | none | The image is too blurry and murky to determine if marine growth is present. |
| F23 | fish | (6.4, 1.8, -0.5) | 30 x 19 cm | 0.34 | 3 | kf_0158, kf_0159, kf_0160 | info | The shape is indistinct and cannot be confidently identified as a fish. |
| F24 | window | n/a (no pose or depth) | n/a | 0.34 | 1 | kf_0119 | info | The image is heavily obscured by water turbidity, making it impossible to confirm if any structure is a window. |
| F25 | hole_damage | (5.6, 9.0, 0.8) | 9 x 13 cm | 0.34 | 6 | kf_0042, kf_0043, kf_0044... | none | The image is too blurry to determine if it represents a hole or damage. |
| F26 | crack | n/a (no pose or depth) | n/a | 0.34 | 2 | kf_0172, kf_0173 | none | The line appears to be a minor imperfection or shadow, not a clear crack. |
| F27 | debris | n/a (no pose or depth) | n/a | 0.34 | 1 | kf_0059 | info | The image is indistinct and lacks clear features to identify debris. |
| F28 | fish | (2.8, 5.7, -0.2) | 39 x 27 cm | 0.34 | 17 | kf_0014, kf_0015, kf_0016... | info | The image is too blurry and low-resolution to confirm if it is a fish. |
| F29 | hole_damage | (2.0, -1.3, 0.6) | 19 x 32 cm | 0.34 | 13 | kf_0102, kf_0103, kf_0104... | none | The image shows a dark circular area that may be a void or shadow, but no clear evidence of structural damage is visible. |
| F30 | marine_growth | n/a (no pose or depth) | n/a | 0.33 | 1 | kf_0029 | none | The image is too blurry to determine if it is marine growth or another object. |

Crops: `crops/<ID>.jpg` in the output directory (embedded in the HTML version).

## Faithfulness check

- Report-text sentences checked: 34; hallucinated statements: **0** (unknown ids 0, absent-class mentions 0, id/class mismatches 0)
- Findings not mentioned in the text: none
