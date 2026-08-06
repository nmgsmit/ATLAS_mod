import numpy as np

### RARP Palette ###

# Object id 0 is always background ("None") and is never painted: both overlay paths
# in gui/interactive_utils.py compose only where (mask > 0), so id 0 stays as the raw
# frame. It is deliberately absent from custom_names -- the GUI class list is built
# from range(1, num_objects + 1), so background is not a selectable class.
#
# Colours are chosen to survive the *1.5 visualisation boost in interactive_utils.py
# (already-saturated channels clip back to themselves), so the GUI swatch and the
# on-frame overlay show the same hue.

color_palette = {
    1: (255, 255, 0),    # Urethra - Yellow (urine)
    2: (255, 0, 255),    # Prostate - Magenta
    3: (0, 0, 255),      # Dorsal venous plexus - Blue (venous)
    4: (0, 255, 0),      # Catheter - Green (foreign body, contrasts with tissue)
    5: (128, 128, 128),  # Non-anatomical - Gray (neutral, not tissue)
}

custom_names = {
    1: "Urethra",
    2: "Prostate",
    3: "Dorsal venous plexus",
    4: "Catheter",
    5: "Non-anatomical",
}

custom_palette_np = np.array([color_palette.get(i, (0, 0, 0)) for i in range(len(custom_names)+1)])
custom_palette = custom_palette_np.astype(np.uint8).tobytes()
