# Underwater inspection report (draft)

> **AI-GENERATED DRAFT - detections come from an open-vocabulary detector and text from a vision-language model. Severity values are suggestions only and MUST be reviewed by a qualified human inspector.**

## Summary

This automatic draft reports 10 confirmed findings from an underwater inspection covering 585 seconds and 65 meters of track. Confirmed findings include [F01], [F02], [F04], [F05], [F07], [F10], [F11], [F12], [F14], [F15], [F16], [F18], [F19], [F21], [F22], [F24], [F27], [F28], [F29], and [F30]. These are subject to human review for validation, as the system’s output requires manual confirmation due to low visibility and ambiguous visual cues. The inspection detected marine growth, tires, fish, windows, debris, and structural damage, with varying confidence levels. No further detector candidates are included in this summary.

## Coverage

- Source: `bus_outside`
- Duration: 585 s, keyframes: 198 (198 with pose)
- Track length: 64.5 m (VIO trajectory: `/home/nevle.nehal/uw-vslam-bench/results/bus_outside/uwct/s4_rep.tum`)
- Detector: google/owlv2-base-patch16-ensemble; VLM: Qwen/Qwen3-VL-4B-Instruct (context verifier); score threshold 0.15 (base prompts)

## Findings

Positions are in the VIO world frame (metres, start-camera frame, not gravity aligned). Sizes are approximate box extents from stereo depth. Frame refs are keyframe ids.

| ID | Type | Status | Location (x,y,z m) | Size | Conf. | Obs. | Frames | Severity (AI suggestion) | Assessment |
|---|---|---|---|---|---|---|---|---|---|
| F01 | tire | confirmed | (4.2, 0.5, 0.1) | 63 x 95 cm | 0.72 | 10 | kf_0080, kf_0081, kf_0082... | info | The tire appears to be partially submerged in murky water with visible surface texture and some wear or debris. |
| F02 | fish | confirmed | (5.7, 9.1, 1.0) | 28 x 21 cm | 0.62 | 5 | kf_0040, kf_0041, kf_0043... | info | The fish is discernible as a dark, elongated shape moving through the greenish, hazy water. |
| F04 | tire | confirmed | (3.3, 7.2, 0.1) | 86 x 127 cm | 0.44 | 1 | kf_0026 | info | The object is obscured by murkiness and appears heavily corroded or encrusted. |
| F05 | marine_growth | confirmed | (3.6, -1.2, 0.3) | 63 x 129 cm | 0.44 | 20 | kf_0088, kf_0089, kf_0090... | medium | The surface is covered with a dense, irregular pattern of white and dark areas, consistent with biofouling. |
| F07 | marine_growth | confirmed | (3.1, -2.6, -0.5) | 217 x 86 cm | 0.42 | 9 | kf_0093, kf_0094, kf_0095... | low | The surface appears covered in marine growth, with patches of white and dark areas suggesting biofouling. |
| F10 | marine_growth | confirmed | (4.7, -1.4, -0.4) | 209 x 85 cm | 0.40 | 16 | kf_0086, kf_0087, kf_0088... | medium | The surface appears covered in a dense, fuzzy growth consistent with marine biofouling. |
| F11 | marine_growth | confirmed | (2.2, -1.2, 0.2) | 58 x 149 cm | 0.39 | 3 | kf_0180, kf_0182, kf_0183 | low | The surface is heavily encrusted with what appears to be marine growth, including barnacles and sediment, giving it a rough, textured appearance. |
| F12 | window | confirmed | (4.8, 5.4, 1.0) | 279 x 334 cm | 0.38 | 9 | kf_0140, kf_0141, kf_0142... | info | The window is obscured by sediment and low visibility, but its rectangular shape and arrangement are discernible. |
| F14 | marine_growth | confirmed | (2.3, -2.9, -0.5) | 90 x 35 cm | 0.38 | 7 | kf_0180, kf_0181, kf_0181... | medium | The surface appears covered in a dense, irregular growth consistent with marine organisms. |
| F15 | window | confirmed | (4.9, 4.6, 0.8) | 170 x 236 cm | 0.37 | 3 | kf_0067, kf_0067, kf_0068 | info | The window frame appears corroded and covered in marine growth or sediment. |
| F16 | marine_growth | confirmed | (6.7, 4.9, -0.6) | 247 x 87 cm | 0.36 | 2 | kf_0156, kf_0157 | low | The surface appears covered in a dense, fuzzy growth consistent with marine algae or biofouling. |
| F18 | marine_growth | confirmed | (4.6, 3.2, 0.3) | 117 x 138 cm | 0.36 | 5 | kf_0072, kf_0073, kf_0075... | medium | The surface is heavily encrusted with what appears to be marine growth, including algae and possibly barnacles, under murky water conditions. |
| F19 | marine_growth | confirmed | (5.7, 3.6, -0.4) | 119 x 71 cm | 0.35 | 6 | kf_0158, kf_0159, kf_0160... | low | The surface is covered in dense, greenish marine growth, likely algae or biofouling, with some rocks visible. |
| F21 | debris | confirmed | n/a (no pose or depth) | n/a | 0.35 | 1 | kf_0029 | info | The object appears to be a piece of debris with a somewhat rounded, possibly organic or corroded form. |
| F22 | marine_growth | confirmed | (5.1, -0.6, -0.5) | 215 x 96 cm | 0.34 | 4 | kf_0168, kf_0169, kf_0169... | medium | The surface appears covered in a dense, fuzzy growth consistent with marine growth. |
| F24 | window | confirmed | n/a (no pose or depth) | n/a | 0.34 | 1 | kf_0119 | info | The window is obscured by murky water and sediment, with visible greenish light filtering through. |
| F27 | debris | confirmed | n/a (no pose or depth) | n/a | 0.34 | 1 | kf_0059 | info | The object appears to be a small, dark, irregularly shaped object lying on the seafloor. |
| F28 | fish | confirmed | (2.8, 5.7, -0.2) | 39 x 27 cm | 0.34 | 17 | kf_0014, kf_0015, kf_0016... | info | The fish is visible as a blurred, elongated shape moving through the green-tinted water. |
| F29 | hole_damage | confirmed | (2.0, -1.3, 0.6) | 19 x 32 cm | 0.34 | 13 | kf_0102, kf_0103, kf_0104... | medium | The region shows a dark, circular opening that appears to be a hole or damaged section of the metal structure. |
| F30 | marine_growth | confirmed | n/a (no pose or depth) | n/a | 0.33 | 1 | kf_0029 | medium | The growth appears dense and irregular, consistent with marine biofouling. |
| F03 | fish | unverified (VLM unclear/unparsed) | (4.0, 10.3, -0.4) | 15 x 17 cm | 0.48 | 27 | kf_0030, kf_0030, kf_0030... | n/a (not confirmed) | The shape is too indistinct to confidently identify as a fish due to water clarity. |
| F06 | window | unverified (VLM unclear/unparsed) | (2.0, -0.2, 0.3) | 106 x 165 cm | 0.43 | 12 | kf_0000, kf_0001, kf_0002... | n/a (not confirmed) | The surface is obscured by murky water and appears to be covered in marine growth or sediment. |
| F08 | marine_growth | unverified (VLM unclear/unparsed) | (3.2, 9.8, -0.4) | 84 x 48 cm | 0.41 | 15 | kf_0030, kf_0031, kf_0031... | n/a (not confirmed) | The visibility is poor due to murkiness, making it hard to distinguish marine growth from other debris or sediment. |
| F09 | fish | unverified (VLM unclear/unparsed) | (2.0, -1.3, 0.1) | 16 x 24 cm | 0.41 | 2 | kf_0109, kf_0183 | n/a (not confirmed) | The shape is indistinct due to low visibility and murky water, making it difficult to confirm if it is a fish. |
| F13 | fish | unverified (VLM unclear/unparsed) | n/a (no pose or depth) | n/a | 0.38 | 1 | kf_0029 | n/a (not confirmed) | The object is indistinct and lacks clear features to confirm it is a fish. |
| F17 | crack | unverified (VLM unclear/unparsed) | n/a (no pose or depth) | n/a | 0.36 | 2 | kf_0169, kf_0170 | n/a (not confirmed) | The line is faint and indistinct due to water turbidity, making it hard to determine if it is a crack or just a shadow or debris. |
| F20 | crack | unverified (VLM unclear/unparsed) | (2.0, -1.6, -0.1) | 76 x 29 cm | 0.35 | 16 | kf_0099, kf_0102, kf_0103... | n/a (not confirmed) | The line is faint and indistinct, making it difficult to determine if it is a crack or another feature. |
| F23 | fish | unverified (VLM unclear/unparsed) | (6.4, 1.8, -0.5) | 30 x 19 cm | 0.34 | 3 | kf_0158, kf_0159, kf_0160 | n/a (not confirmed) | The shape is too blurry to identify as a fish due to water turbidity. |
| F25 | hole_damage | unverified (VLM unclear/unparsed) | (5.6, 9.0, 0.8) | 9 x 13 cm | 0.34 | 6 | kf_0042, kf_0043, kf_0044... | n/a (not confirmed) | The object's shape and texture are indistinct due to low light and murky water. |
| F26 | crack | unverified (VLM unclear/unparsed) | n/a (no pose or depth) | n/a | 0.34 | 2 | kf_0172, kf_0173 | n/a (not confirmed) | The line is faint and indistinct against the murky background, making it hard to determine if it's a crack or just a shadow or artifact. |

Crops: `crops/<ID>.jpg` in the output directory (embedded in the HTML version).

## Faithfulness check

- Report-text sentences checked: 25; hallucinated statements: **5** (unknown ids 0, absent-class mentions 4, id/class mismatches 0, cites of unconfirmed findings 0)
- Findings not mentioned in the text: none
