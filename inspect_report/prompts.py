"""Inspection classes and OWLv2 text prompts.

'base' = first prompt of each class. 'ens' = all prompts of the class, score = max over prompts.
DEFECT classes get a severity suggestion; the rest are context objects.
"""

CLASS_PROMPTS: dict[str, list[str]] = {
    "corrosion": ["rust", "corrosion on metal", "rusty corroded surface"],
    "marine_growth": ["marine growth", "algae or biofouling on a surface", "barnacles and sediment growth"],
    "crack": ["crack", "a crack in a surface", "fracture line"],
    "hole_damage": ["hole", "damaged or broken part", "a hole in metal"],
    "debris": ["debris", "trash or litter", "junk lying on the bottom"],
    "fish": ["fish", "a fish swimming", "school of fish"],
    "rope_cable": ["rope", "cable", "a rope or cable hanging"],
    "metal_structure": ["metal structure", "steel frame", "a vehicle body or hull"],
    "window": ["window", "a window frame", "glass window"],
    "tire": ["tire", "a car wheel", "rubber tyre"],
}
DEFECT_CLASSES = ["corrosion", "marine_growth", "crack", "hole_damage"]
CLASSES = list(CLASS_PROMPTS)

# lexicon used by the faithfulness check: text mention -> class
CLASS_TERMS: dict[str, list[str]] = {
    "corrosion": ["corrosion", "corroded", "rust"],
    "marine_growth": ["marine growth", "biofouling", "algae", "barnacle"],
    "crack": ["crack"],
    "hole_damage": ["hole", "damage", "breach"],
    "debris": ["debris", "litter", "trash"],
    "fish": ["fish"],
    "rope_cable": ["rope", "cable"],
    "metal_structure": ["metal structure", "steel structure", "hull", "metal frame"],
    "window": ["window"],
    "tire": ["tire", "tyre", "wheel"],
}
