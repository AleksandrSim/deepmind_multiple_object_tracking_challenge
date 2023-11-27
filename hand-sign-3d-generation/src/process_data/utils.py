import json
import os

import bpy
import numpy as np



HAND_BONES = [
    'RightFinger1Metacarpal', 'RightFinger1Proximal', 'RightFinger1Distal', 'RightFinger1Tip', 
    'RightFinger2Metacarpal', 'RightFinger2Proximal', 'RightFinger2Medial', 'RightFinger2Distal', 'RightFinger2Tip', 
    'RightFinger3Metacarpal', 'RightFinger3Proximal', 'RightFinger3Medial', 'RightFinger3Distal', 'RightFinger3Tip', 
    'RightFinger4Metacarpal', 'RightFinger4Proximal', 'RightFinger4Medial', 'RightFinger4Distal', 'RightFinger4Tip', 
    'RightFinger5Metacarpal', 'RightFinger5Proximal', 'RightFinger5Medial', 'RightFinger5Distal', 'RightFinger5Tip'
]

HAND_BONES_CONNECTIONS = [
    ('RightFinger1Metacarpal', 'RightFinger1Proximal'),
    ('RightFinger1Proximal', 'RightFinger1Distal'),
    ('RightFinger1Distal', 'RightFinger1Tip'),
    
    ('RightFinger2Metacarpal', 'RightFinger2Proximal'),
    ('RightFinger2Proximal', 'RightFinger2Medial'),
    ('RightFinger2Medial', 'RightFinger2Distal'),
    ('RightFinger2Distal', 'RightFinger2Tip'),
    
    ('RightFinger3Metacarpal', 'RightFinger3Proximal'),
    ('RightFinger3Proximal', 'RightFinger3Medial'),
    ('RightFinger3Medial', 'RightFinger3Distal'),
    ('RightFinger3Distal', 'RightFinger3Tip'),
    
    ('RightFinger4Metacarpal', 'RightFinger4Proximal'),
    ('RightFinger4Proximal', 'RightFinger4Medial'),
    ('RightFinger4Medial', 'RightFinger4Distal'),
    ('RightFinger4Distal', 'RightFinger4Tip'),
    
    ('RightFinger5Metacarpal', 'RightFinger5Proximal'),
    ('RightFinger5Proximal', 'RightFinger5Medial'),
    ('RightFinger5Medial', 'RightFinger5Distal'),
    ('RightFinger5Distal', 'RightFinger5Tip')
]


cyrillic_to_latin_mapping = {
    "А": "A",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Е": "E",
    "Ё": "YO",
    "Ж": "Zh",
    "З": "Z",
    "И": "I",
    "Й": "Y",
    "К": "K",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ф": "F",
    "Х": "H",
    "Ц": "Ts",
    "Ч": "Ch",
    "Ш": "Sh",
    "Щ": "Shch",
    "Ъ": "HARD",  # No direct equivalent
    "Ы": "Y",
    "Ь": "SOFT",  # No direct equivalent
    "Э": "E",
    "Ю": "Yu",
    "Я": "Ya",
    # The following Cyrillic letters do not have direct Latin equivalents:
    # Ъ, Ь
}

latin_to_cyrillic_mapping = {value: key for key, value in cyrillic_to_latin_mapping.items()}
