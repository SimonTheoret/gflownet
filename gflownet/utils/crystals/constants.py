ELEMENT_NAMES = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    113: "Nh",
    114: "Fl",
    115: "Mc",
    116: "Lv",
    117: "Ts",
    118: "Og",
}

OXIDATION_STATES = {
    1: [-1, 1],
    2: [0],
    3: [0, 1],
    4: [0, 1, 2],
    5: [-5, -1, 0, 1, 2, 3],
    6: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    7: [-3, -2, -1, 0, 1, 2, 3, 4, 5],
    8: [-2, -1, 0, 1, 2],
    9: [-1, 0],
    10: [0],
    11: [-1, 0, 1],
    12: [0, 1, 2],
    13: [-2, -1, 0, 1, 2, 3],
    14: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    15: [-3, -2, -1, 0, 1, 2, 3, 4, 5],
    16: [-2, -1, 0, 1, 2, 3, 4, 5, 6],
    17: [-1, 1, 2, 3, 4, 5, 6, 7],
    18: [0],
    19: [-1, 1],
    20: [1, 2],
    21: [0, 1, 2, 3],
    22: [-2, -1, 0, 1, 2, 3, 4],
    23: [-3, -1, 0, 1, 2, 3, 4, 5],
    24: [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6],
    25: [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    26: [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    27: [-3, -1, 0, 1, 2, 3, 4, 5],
    28: [-2, -1, 0, 1, 2, 3, 4],
    29: [-2, 0, 1, 2, 3, 4],
    30: [-2, 0, 1, 2],
    31: [-5, -4, -3, -2, -1, 0, 1, 2, 3],
    32: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    33: [-3, -2, -1, 0, 1, 2, 3, 4, 5],
    34: [-2, -1, 0, 1, 2, 3, 4, 5, 6],
    35: [-1, 1, 2, 3, 4, 5, 7],
    36: [0, 1, 2],
    37: [-1, 1],
    38: [1, 2],
    39: [0, 1, 2, 3],
    40: [-2, 0, 1, 2, 3, 4],
    41: [-3, -1, 0, 1, 2, 3, 4, 5],
    42: [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6],
    43: [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    44: [-4, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    45: [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    46: [0, 1, 2, 3, 4, 5],
    47: [-2, -1, 0, 1, 2, 3],
    48: [-2, 1, 2],
    49: [-5, -2, -1, 0, 1, 2, 3],
    50: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    51: [-3, -2, -1, 0, 1, 2, 3, 4, 5],
    52: [-2, -1, 0, 1, 2, 3, 4, 5, 6],
    53: [-1, 1, 2, 3, 4, 5, 6, 7],
    54: [0, 2, 4, 6, 8],
    55: [-1, 1],
    56: [1, 2],
    57: [0, 1, 2, 3],
    58: [2, 3, 4],
    59: [0, 1, 2, 3, 4, 5],
    60: [0, 2, 3, 4],
    61: [2, 3],
    62: [0, 1, 2, 3],
    63: [0, 2, 3],
    64: [0, 1, 2, 3],
    65: [0, 1, 2, 3, 4],
    66: [0, 2, 3, 4],
    67: [0, 2, 3],
    68: [0, 2, 3],
    69: [0, 1, 2, 3],
    70: [0, 1, 2, 3],
    71: [0, 2, 3],
    72: [-2, 0, 1, 2, 3, 4],
    73: [-3, -1, 0, 1, 2, 3, 4, 5],
    74: [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6],
    75: [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7],
    76: [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    77: [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    78: [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
    79: [-3, -2, -1, 0, 1, 2, 3, 5],
    80: [-2, 1, 2],
    81: [-5, -2, -1, 1, 2, 3],
    82: [-4, -2, -1, 0, 1, 2, 3, 4],
    83: [-3, -2, -1, 0, 1, 2, 3, 4, 5],
    84: [-2, 2, 4, 5, 6],
    85: [-1, 1, 3, 5, 7],
    86: [2, 6],
    87: [1],
    88: [2],
    89: [2, 3],
    90: [-1, 1, 2, 3, 4],
    91: [2, 3, 4, 5],
    92: [-1, 1, 2, 3, 4, 5, 6],
    93: [2, 3, 4, 5, 6, 7],
    94: [2, 3, 4, 5, 6, 7, 8],
    95: [2, 3, 4, 5, 6, 7],
    96: [3, 4, 5, 6],
    97: [2, 3, 4, 5],
    98: [2, 3, 4, 5],
    99: [2, 3, 4],
    100: [2, 3],
    101: [2, 3],
    102: [2, 3],
    103: [3],
    104: [4],
    105: [5],
    106: [0, 6],
    107: [7],
    108: [8],
    109: [0],
    110: [0],
    111: [0],
    112: [2],
    113: [0],
    114: [0],
    115: [0],
    116: [0],
    117: [0],
    118: [0],
}

CUBIC = "cubic"
HEXAGONAL = "hexagonal"
MONOCLINIC = "monoclinic"
ORTHORHOMBIC = "orthorhombic"
RHOMBOHEDRAL = "rhombohedral"
TETRAGONAL = "tetragonal"
TRICLINIC = "triclinic"

LATTICE_SYSTEMS = [
    CUBIC,
    HEXAGONAL,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
]

# See: https://en.wikipedia.org/wiki/Crystal_system#Crystal_classes
# Each item in the dictionary contains a list of:
# - crystal system name
# - crystal class indices in the crystal system
# - point symmetry indices in the crystal system
CRYSTAL_SYSTEMS = {
    1: ['triclinic', [1, 2], [1, 2]],
    2: ['monoclinic', [3, 4, 5], [1, 3, 2]],
    3: ['orthorhombic', [6, 7, 8], [4, 3, 2]],
    4: ['tetragonal', [9, 10, 11, 12, 13, 14, 15], [1, 5, 2, 4, 3]],
    5: ['trigonal', [16, 17, 18, 19, 20], [1, 2, 4, 3]],
    6: ['hexagonal', [21, 22, 23, 24, 25, 26, 27], [1, 5, 2, 4, 3]],
    7: ['cubic', [28, 29, 30, 31, 32], [4, 2, 5]],
}
CRYSTAL_SYSTEMS_MINIMAL = {
    1: "triclinic",
    2: "monoclinic",
    3: "orthorhombic",
    4: "tetragonal",
    5: "trigonal",
    6: "hexagonal",
    7: "cubic",
}

# See: https://en.wikipedia.org/wiki/Crystal_system#Crystal_classes
# See: http://pd.chem.ucl.ac.uk/pdnn/symm2/group32.htm
# Each item in the dictionary contains a list of:
# - crystal class name
# - crystal system
# - list of point groups as in pymatgen
# - point symmetry
# - space groups
CRYSTAL_CLASSES = {
    1: ["pedial", "triclinic", ["1"], "enantiomorphic-polar", [1]],
    2: ["pinacoidal", "triclinic", ["-1"], "centrosymmetric", [2]],
    3: ["sphenoidal", "monoclinic", ["2"], "enantiomorphic-polar", [3, 4, 5]],
    4: ["domatic", "monoclinic", ["m"], "polar", [6, 7, 8, 9]],
    5: [
        "prismatic",
        "monoclinic",
        ["2/m"],
        "centrosymmetric",
        [10, 11, 12, 13, 14, 15],
    ],
    6: [
        "rhombic-disphenoidal",
        "orthorhombic",
        ["222"],
        "enantiomorphic",
        [16, 17, 18, 19, 20, 21, 22, 23, 24],
    ],
    7: [
        "rhombic-pyramidal",
        "orthorhombic",
        ["mm2"],
        "polar",
        [
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
        ],
    ],
    8: [
        "rhombic-dipyramidal",
        "orthorhombic",
        ["mmm"],
        "centrosymmetric",
        [
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
        ],
    ],
    9: [
        "tetragonal-pyramidal",
        "tetragonal",
        ["4"],
        "enantiomorphic-polar",
        [75, 76, 77, 78, 79, 80],
    ],
    10: [
        "tetragonal-disphenoidal",
        "tetragonal",
        ["-4"],
        "non-centrosymmetric",
        [81, 82],
    ],
    11: [
        "tetragonal-dipyramidal",
        "tetragonal",
        ["4/m"],
        "centrosymmetric",
        [83, 84, 85, 86, 87, 88],
    ],
    12: [
        "tetragonal-trapezohedral",
        "tetragonal",
        ["422"],
        "enantiomorphic",
        [89, 90, 91, 92, 93, 94, 95, 96, 97, 98],
    ],
    13: [
        "ditetragonal-pyramidal",
        "tetragonal",
        ["4mm"],
        "polar",
        [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    ],
    14: [
        "tetragonal-scalenohedral",
        "tetragonal",
        ["-42m", "-4m2"],
        "non-centrosymmetric",
        [111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
    ],
    15: [
        "ditetragonal-dipyramidal",
        "tetragonal",
        ["4/mmm"],
        "centrosymmetric",
        [
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
        ],
    ],
    16: [
        "trigonal-pyramidal",
        "trigonal",
        ["3"],
        "enantiomorphic-polar",
        [143, 144, 145, 146],
    ],
    17: ["rhombohedral", "trigonal", ["-3"], "centrosymmetric", [147, 148]],
    18: [
        "trigonal-trapezohedral",
        "trigonal",
        ["32", "321", "312"],
        "enantiomorphic",
        [149, 150, 151, 152, 153, 154, 155],
    ],
    19: [
        "ditrigonal-pyramidal",
        "trigonal",
        ["3m", "3m1", "31m"],
        "polar",
        [156, 157, 158, 159, 160, 161],
    ],
    20: [
        "ditrigonal-scalenohedral",
        "trigonal",
        ["-3m", "-3m1", "-31m"],
        "centrosymmetric",
        [162, 163, 164, 165, 166, 167],
    ],
    21: [
        "hexagonal-pyramidal",
        "hexagonal",
        ["6"],
        "enantiomorphic-polar",
        [168, 169, 170, 171, 172, 173],
    ],
    22: ["trigonal-dipyramidal", "hexagonal", ["-6"], "non-centrosymmetric", [174]],
    23: [
        "hexagonal-dipyramidal",
        "hexagonal",
        ["6/m"],
        "centrosymmetric",
        [175, 176],
    ],
    24: [
        "hexagonal-trapezohedral",
        "hexagonal",
        ["622"],
        "enantiomorphic",
        [177, 178, 179, 180, 181, 182],
    ],
    25: [
        "dihexagonal-pyramidal",
        "hexagonal",
        ["6mm"],
        "polar",
        [183, 184, 185, 186],
    ],
    26: [
        "ditrigonal-dipyramidal",
        "hexagonal",
        ["-6m2", "-62m"],
        "non-centrosymmetric",
        [187, 188, 189, 190],
    ],
    27: [
        "dihexagonal-dipyramidal",
        "hexagonal",
        ["6/mmm"],
        "centrosymmetric",
        [191, 192, 193, 194],
    ],
    28: ["tetartoidal", "cubic", ["23"], "enantiomorphic", [195, 196, 197, 198, 199]],
    29: [
        "diploidal",
        "cubic",
        ["m-3"],
        "centrosymmetric",
        [200, 201, 202, 203, 204, 205, 206],
    ],
    30: [
        "gyroidal",
        "cubic",
        ["432"],
        "enantiomorphic",
        [207, 208, 209, 210, 211, 212, 213, 214],
    ],
    31: [
        "hextetrahedral",
        "cubic",
        ["-43m"],
        "non-centrosymmetric",
        [215, 216, 217, 218, 219, 220],
    ],
    32: [
        "hexoctahedral",
        "cubic",
        ["m-3m"],
        "centrosymmetric",
        [221, 222, 223, 224, 225, 226, 227, 228, 229, 230],
    ],
}
CRYSTAL_CLASSES_NOSG = {
    1: ["pedial", "triclinic", ["1"], "enantiomorphic-polar"],
    2: ["pinacoidal", "triclinic", ["-1"], "centrosymmetric"],
    3: ["sphenoidal", "monoclinic", ["2"], "enantiomorphic-polar"],
    4: ["domatic", "monoclinic", ["m"], "polar"],
    5: ["prismatic", "monoclinic", ["2/m"], "centrosymmetric"],
    6: ["rhombic-disphenoidal", "orthorhombic", ["222"], "enantiomorphic"],
    7: ["rhombic-pyramidal", "orthorhombic", ["mm2"], "polar"],
    8: ["rhombic-dipyramidal", "orthorhombic", ["mmm"], "centrosymmetric"],
    9: ["tetragonal-pyramidal", "tetragonal", ["4"], "enantiomorphic-polar"],
    10: ["tetragonal-disphenoidal", "tetragonal", ["-4"], "non-centrosymmetric"],
    11: ["tetragonal-dipyramidal", "tetragonal", ["4/m"], "centrosymmetric"],
    12: ["tetragonal-trapezohedral", "tetragonal", ["422"], "enantiomorphic"],
    13: ["ditetragonal-pyramidal", "tetragonal", ["4mm"], "polar"],
    14: [
        "tetragonal-scalenohedral",
        "tetragonal",
        ["-42m", "-4m2"],
        "non-centrosymmetric",
    ],
    15: ["ditetragonal-dipyramidal", "tetragonal", ["4/mmm"], "centrosymmetric"],
    16: ["trigonal-pyramidal", "trigonal", ["3"], "enantiomorphic-polar"],
    17: ["rhombohedral", "trigonal", ["-3"], "centrosymmetric"],
    18: [
        "trigonal-trapezohedral",
        "trigonal",
        ["32", "321", "312"],
        "enantiomorphic",
    ],
    19: ["ditrigonal-pyramidal", "trigonal", ["3m", "3m1", "31m"], "polar"],
    20: [
        "ditrigonal-scalenohedral",
        "trigonal",
        ["-3m", "-3m1", "-31m"],
        "centrosymmetric",
    ],
    21: ["hexagonal-pyramidal", "hexagonal", ["6"], "enantiomorphic-polar"],
    22: ["trigonal-dipyramidal", "hexagonal", ["-6"], "non-centrosymmetric"],
    23: ["hexagonal-dipyramidal", "hexagonal", ["6/m"], "centrosymmetric"],
    24: ["hexagonal-trapezohedral", "hexagonal", ["622"], "enantiomorphic"],
    25: ["dihexagonal-pyramidal", "hexagonal", ["6mm"], "polar"],
    26: [
        "ditrigonal-dipyramidal",
        "hexagonal",
        ["-6m2", "-62m"],
        "non-centrosymmetric",
    ],
    27: ["dihexagonal-dipyramidal", "hexagonal", ["6/mmm"], "centrosymmetric"],
    28: ["tetartoidal", "cubic", ["23"], "enantiomorphic"],
    29: ["diploidal", "cubic", ["m-3"], "centrosymmetric"],
    30: ["gyroidal", "cubic", ["432"], "enantiomorphic"],
    31: ["hextetrahedral", "cubic", ["-43m"], "non-centrosymmetric"],
    32: ["hexoctahedral", "cubic", ["m-3m"], "centrosymmetric"],
}

# See: https://en.wikipedia.org/wiki/Crystal_system#Crystal_classes
# See: http://pd.chem.ucl.ac.uk/pdnn/symm2/group32.htm
# Each item in the dictionary contains a list of:
# - point symmetry name
# - crystal system indices with the point symmetry
# - crystal class indices with the point symmetry
POINT_SYMMETRIES = {
    1: ['enantiomorphic-polar', [1, 2, 4, 5, 6], [1, 3, 9, 16, 21]],
    2: ['centrosymmetric', [1, 2, 3, 4, 5, 6, 7], [2, 5, 8, 11, 15, 17, 20, 23, 27, 29, 32]],
    3: ['polar', [2, 3, 4, 5, 6], [4, 7, 13, 19, 25]],
    4: ['enantiomorphic', [3, 4, 5, 6, 7], [6, 12, 18, 24, 28, 30]],
    5: ['non-centrosymmetric', [4, 6, 7], [10, 14, 22, 26, 31]],
}
POINT_SYMMETRIES_MINIMAL = {
    1: "enantiomorphic-polar",
    2: "centrosymmetric",
    3: "polar",
    4: "enantiomorphic",
    5: "non-centrosymmetric",
}

# Point group of each space group, from pymatgen
# See: ./scripts/pymatgen/pg_of_all_sg.py
# Each item in the dictionary contains a list of:
# - point group as in pymatgen
# - crystal class index of the point group
# - crystal system index
# - point symmetry index
SPACE_GROUPS = {
    1: ['1', 1, 1, 1],
    2: ['-1', 2, 1, 2],
    3: ['2', 3, 2, 1],
    4: ['2', 3, 2, 1],
    5: ['2', 3, 2, 1],
    6: ['m', 4, 2, 3],
    7: ['m', 4, 2, 3],
    8: ['m', 4, 2, 3],
    9: ['m', 4, 2, 3],
    10: ['2/m', 5, 2, 2],
    11: ['2/m', 5, 2, 2],
    12: ['2/m', 5, 2, 2],
    13: ['2/m', 5, 2, 2],
    14: ['2/m', 5, 2, 2],
    15: ['2/m', 5, 2, 2],
    16: ['222', 6, 3, 4],
    17: ['222', 6, 3, 4],
    18: ['222', 6, 3, 4],
    19: ['222', 6, 3, 4],
    20: ['222', 6, 3, 4],
    21: ['222', 6, 3, 4],
    22: ['222', 6, 3, 4],
    23: ['222', 6, 3, 4],
    24: ['222', 6, 3, 4],
    25: ['mm2', 7, 3, 3],
    26: ['mm2', 7, 3, 3],
    27: ['mm2', 7, 3, 3],
    28: ['mm2', 7, 3, 3],
    29: ['mm2', 7, 3, 3],
    30: ['mm2', 7, 3, 3],
    31: ['mm2', 7, 3, 3],
    32: ['mm2', 7, 3, 3],
    33: ['mm2', 7, 3, 3],
    34: ['mm2', 7, 3, 3],
    35: ['mm2', 7, 3, 3],
    36: ['mm2', 7, 3, 3],
    37: ['mm2', 7, 3, 3],
    38: ['mm2', 7, 3, 3],
    39: ['mm2', 7, 3, 3],
    40: ['mm2', 7, 3, 3],
    41: ['mm2', 7, 3, 3],
    42: ['mm2', 7, 3, 3],
    43: ['mm2', 7, 3, 3],
    44: ['mm2', 7, 3, 3],
    45: ['mm2', 7, 3, 3],
    46: ['mm2', 7, 3, 3],
    47: ['mmm', 8, 3, 2],
    48: ['mmm', 8, 3, 2],
    49: ['mmm', 8, 3, 2],
    50: ['mmm', 8, 3, 2],
    51: ['mmm', 8, 3, 2],
    52: ['mmm', 8, 3, 2],
    53: ['mmm', 8, 3, 2],
    54: ['mmm', 8, 3, 2],
    55: ['mmm', 8, 3, 2],
    56: ['mmm', 8, 3, 2],
    57: ['mmm', 8, 3, 2],
    58: ['mmm', 8, 3, 2],
    59: ['mmm', 8, 3, 2],
    60: ['mmm', 8, 3, 2],
    61: ['mmm', 8, 3, 2],
    62: ['mmm', 8, 3, 2],
    63: ['mmm', 8, 3, 2],
    64: ['mmm', 8, 3, 2],
    65: ['mmm', 8, 3, 2],
    66: ['mmm', 8, 3, 2],
    67: ['mmm', 8, 3, 2],
    68: ['mmm', 8, 3, 2],
    69: ['mmm', 8, 3, 2],
    70: ['mmm', 8, 3, 2],
    71: ['mmm', 8, 3, 2],
    72: ['mmm', 8, 3, 2],
    73: ['mmm', 8, 3, 2],
    74: ['mmm', 8, 3, 2],
    75: ['4', 9, 4, 1],
    76: ['4', 9, 4, 1],
    77: ['4', 9, 4, 1],
    78: ['4', 9, 4, 1],
    79: ['4', 9, 4, 1],
    80: ['4', 9, 4, 1],
    81: ['-4', 10, 4, 5],
    82: ['-4', 10, 4, 5],
    83: ['4/m', 11, 4, 2],
    84: ['4/m', 11, 4, 2],
    85: ['4/m', 11, 4, 2],
    86: ['4/m', 11, 4, 2],
    87: ['4/m', 11, 4, 2],
    88: ['4/m', 11, 4, 2],
    89: ['422', 12, 4, 4],
    90: ['422', 12, 4, 4],
    91: ['422', 12, 4, 4],
    92: ['422', 12, 4, 4],
    93: ['422', 12, 4, 4],
    94: ['422', 12, 4, 4],
    95: ['422', 12, 4, 4],
    96: ['422', 12, 4, 4],
    97: ['422', 12, 4, 4],
    98: ['422', 12, 4, 4],
    99: ['4mm', 13, 4, 3],
    100: ['4mm', 13, 4, 3],
    101: ['4mm', 13, 4, 3],
    102: ['4mm', 13, 4, 3],
    103: ['4mm', 13, 4, 3],
    104: ['4mm', 13, 4, 3],
    105: ['4mm', 13, 4, 3],
    106: ['4mm', 13, 4, 3],
    107: ['4mm', 13, 4, 3],
    108: ['4mm', 13, 4, 3],
    109: ['4mm', 13, 4, 3],
    110: ['4mm', 13, 4, 3],
    111: ['-42m', 14, 4, 5],
    112: ['-42m', 14, 4, 5],
    113: ['-42m', 14, 4, 5],
    114: ['-42m', 14, 4, 5],
    115: ['-4m2', 14, 4, 5],
    116: ['-4m2', 14, 4, 5],
    117: ['-4m2', 14, 4, 5],
    118: ['-4m2', 14, 4, 5],
    119: ['-4m2', 14, 4, 5],
    120: ['-4m2', 14, 4, 5],
    121: ['-42m', 14, 4, 5],
    122: ['-42m', 14, 4, 5],
    123: ['4/mmm', 15, 4, 2],
    124: ['4/mmm', 15, 4, 2],
    125: ['4/mmm', 15, 4, 2],
    126: ['4/mmm', 15, 4, 2],
    127: ['4/mmm', 15, 4, 2],
    128: ['4/mmm', 15, 4, 2],
    129: ['4/mmm', 15, 4, 2],
    130: ['4/mmm', 15, 4, 2],
    131: ['4/mmm', 15, 4, 2],
    132: ['4/mmm', 15, 4, 2],
    133: ['4/mmm', 15, 4, 2],
    134: ['4/mmm', 15, 4, 2],
    135: ['4/mmm', 15, 4, 2],
    136: ['4/mmm', 15, 4, 2],
    137: ['4/mmm', 15, 4, 2],
    138: ['4/mmm', 15, 4, 2],
    139: ['4/mmm', 15, 4, 2],
    140: ['4/mmm', 15, 4, 2],
    141: ['4/mmm', 15, 4, 2],
    142: ['4/mmm', 15, 4, 2],
    143: ['3', 16, 5, 1],
    144: ['3', 16, 5, 1],
    145: ['3', 16, 5, 1],
    146: ['3', 16, 5, 1],
    147: ['-3', 17, 5, 2],
    148: ['-3', 17, 5, 2],
    149: ['312', 18, 5, 4],
    150: ['321', 18, 5, 4],
    151: ['312', 18, 5, 4],
    152: ['321', 18, 5, 4],
    153: ['312', 18, 5, 4],
    154: ['321', 18, 5, 4],
    155: ['32', 18, 5, 4],
    156: ['3m1', 19, 5, 3],
    157: ['31m', 19, 5, 3],
    158: ['3m1', 19, 5, 3],
    159: ['31m', 19, 5, 3],
    160: ['3m', 19, 5, 3],
    161: ['3m', 19, 5, 3],
    162: ['-31m', 20, 5, 2],
    163: ['-31m', 20, 5, 2],
    164: ['-3m1', 20, 5, 2],
    165: ['-3m1', 20, 5, 2],
    166: ['-3m', 20, 5, 2],
    167: ['-3m', 20, 5, 2],
    168: ['6', 21, 6, 1],
    169: ['6', 21, 6, 1],
    170: ['6', 21, 6, 1],
    171: ['6', 21, 6, 1],
    172: ['6', 21, 6, 1],
    173: ['6', 21, 6, 1],
    174: ['-6', 22, 6, 5],
    175: ['6/m', 23, 6, 2],
    176: ['6/m', 23, 6, 2],
    177: ['622', 24, 6, 4],
    178: ['622', 24, 6, 4],
    179: ['622', 24, 6, 4],
    180: ['622', 24, 6, 4],
    181: ['622', 24, 6, 4],
    182: ['622', 24, 6, 4],
    183: ['6mm', 25, 6, 3],
    184: ['6mm', 25, 6, 3],
    185: ['6mm', 25, 6, 3],
    186: ['6mm', 25, 6, 3],
    187: ['-6m2', 26, 6, 5],
    188: ['-6m2', 26, 6, 5],
    189: ['-62m', 26, 6, 5],
    190: ['-62m', 26, 6, 5],
    191: ['6/mmm', 27, 6, 2],
    192: ['6/mmm', 27, 6, 2],
    193: ['6/mmm', 27, 6, 2],
    194: ['6/mmm', 27, 6, 2],
    195: ['23', 28, 7, 4],
    196: ['23', 28, 7, 4],
    197: ['23', 28, 7, 4],
    198: ['23', 28, 7, 4],
    199: ['23', 28, 7, 4],
    200: ['m-3', 29, 7, 2],
    201: ['m-3', 29, 7, 2],
    202: ['m-3', 29, 7, 2],
    203: ['m-3', 29, 7, 2],
    204: ['m-3', 29, 7, 2],
    205: ['m-3', 29, 7, 2],
    206: ['m-3', 29, 7, 2],
    207: ['432', 30, 7, 4],
    208: ['432', 30, 7, 4],
    209: ['432', 30, 7, 4],
    210: ['432', 30, 7, 4],
    211: ['432', 30, 7, 4],
    212: ['432', 30, 7, 4],
    213: ['432', 30, 7, 4],
    214: ['432', 30, 7, 4],
    215: ['-43m', 31, 7, 5],
    216: ['-43m', 31, 7, 5],
    217: ['-43m', 31, 7, 5],
    218: ['-43m', 31, 7, 5],
    219: ['-43m', 31, 7, 5],
    220: ['-43m', 31, 7, 5],
    221: ['m-3m', 32, 7, 2],
    222: ['m-3m', 32, 7, 2],
    223: ['m-3m', 32, 7, 2],
    224: ['m-3m', 32, 7, 2],
    225: ['m-3m', 32, 7, 2],
    226: ['m-3m', 32, 7, 2],
    227: ['m-3m', 32, 7, 2],
    228: ['m-3m', 32, 7, 2],
    229: ['m-3m', 32, 7, 2],
    230: ['m-3m', 32, 7, 2],
}
SPACE_GROUPS_MINIMAL = {
    1: "1",
    2: "-1",
    3: "2",
    4: "2",
    5: "2",
    6: "m",
    7: "m",
    8: "m",
    9: "m",
    10: "2/m",
    11: "2/m",
    12: "2/m",
    13: "2/m",
    14: "2/m",
    15: "2/m",
    16: "222",
    17: "222",
    18: "222",
    19: "222",
    20: "222",
    21: "222",
    22: "222",
    23: "222",
    24: "222",
    25: "mm2",
    26: "mm2",
    27: "mm2",
    28: "mm2",
    29: "mm2",
    30: "mm2",
    31: "mm2",
    32: "mm2",
    33: "mm2",
    34: "mm2",
    35: "mm2",
    36: "mm2",
    37: "mm2",
    38: "mm2",
    39: "mm2",
    40: "mm2",
    41: "mm2",
    42: "mm2",
    43: "mm2",
    44: "mm2",
    45: "mm2",
    46: "mm2",
    47: "mmm",
    48: "mmm",
    49: "mmm",
    50: "mmm",
    51: "mmm",
    52: "mmm",
    53: "mmm",
    54: "mmm",
    55: "mmm",
    56: "mmm",
    57: "mmm",
    58: "mmm",
    59: "mmm",
    60: "mmm",
    61: "mmm",
    62: "mmm",
    63: "mmm",
    64: "mmm",
    65: "mmm",
    66: "mmm",
    67: "mmm",
    68: "mmm",
    69: "mmm",
    70: "mmm",
    71: "mmm",
    72: "mmm",
    73: "mmm",
    74: "mmm",
    75: "4",
    76: "4",
    77: "4",
    78: "4",
    79: "4",
    80: "4",
    81: "-4",
    82: "-4",
    83: "4/m",
    84: "4/m",
    85: "4/m",
    86: "4/m",
    87: "4/m",
    88: "4/m",
    89: "422",
    90: "422",
    91: "422",
    92: "422",
    93: "422",
    94: "422",
    95: "422",
    96: "422",
    97: "422",
    98: "422",
    99: "4mm",
    100: "4mm",
    101: "4mm",
    102: "4mm",
    103: "4mm",
    104: "4mm",
    105: "4mm",
    106: "4mm",
    107: "4mm",
    108: "4mm",
    109: "4mm",
    110: "4mm",
    111: "-42m",
    112: "-42m",
    113: "-42m",
    114: "-42m",
    115: "-4m2",
    116: "-4m2",
    117: "-4m2",
    118: "-4m2",
    119: "-4m2",
    120: "-4m2",
    121: "-42m",
    122: "-42m",
    123: "4/mmm",
    124: "4/mmm",
    125: "4/mmm",
    126: "4/mmm",
    127: "4/mmm",
    128: "4/mmm",
    129: "4/mmm",
    130: "4/mmm",
    131: "4/mmm",
    132: "4/mmm",
    133: "4/mmm",
    134: "4/mmm",
    135: "4/mmm",
    136: "4/mmm",
    137: "4/mmm",
    138: "4/mmm",
    139: "4/mmm",
    140: "4/mmm",
    141: "4/mmm",
    142: "4/mmm",
    143: "3",
    144: "3",
    145: "3",
    146: "3",
    147: "-3",
    148: "-3",
    149: "312",
    150: "321",
    151: "312",
    152: "321",
    153: "312",
    154: "321",
    155: "32",
    156: "3m1",
    157: "31m",
    158: "3m1",
    159: "31m",
    160: "3m",
    161: "3m",
    162: "-31m",
    163: "-31m",
    164: "-3m1",
    165: "-3m1",
    166: "-3m",
    167: "-3m",
    168: "6",
    169: "6",
    170: "6",
    171: "6",
    172: "6",
    173: "6",
    174: "-6",
    175: "6/m",
    176: "6/m",
    177: "622",
    178: "622",
    179: "622",
    180: "622",
    181: "622",
    182: "622",
    183: "6mm",
    184: "6mm",
    185: "6mm",
    186: "6mm",
    187: "-6m2",
    188: "-6m2",
    189: "-62m",
    190: "-62m",
    191: "6/mmm",
    192: "6/mmm",
    193: "6/mmm",
    194: "6/mmm",
    195: "23",
    196: "23",
    197: "23",
    198: "23",
    199: "23",
    200: "m-3",
    201: "m-3",
    202: "m-3",
    203: "m-3",
    204: "m-3",
    205: "m-3",
    206: "m-3",
    207: "432",
    208: "432",
    209: "432",
    210: "432",
    211: "432",
    212: "432",
    213: "432",
    214: "432",
    215: "-43m",
    216: "-43m",
    217: "-43m",
    218: "-43m",
    219: "-43m",
    220: "-43m",
    221: "m-3m",
    222: "m-3m",
    223: "m-3m",
    224: "m-3m",
    225: "m-3m",
    226: "m-3m",
    227: "m-3m",
    228: "m-3m",
    229: "m-3m",
    230: "m-3m",
}
