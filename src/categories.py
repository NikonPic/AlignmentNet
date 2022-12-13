# name of all extra pointfiles
SEG_NAMES = [
    'H', 'F', 'T', 'Fi', 'S', 'A', 'O', 'K',
]

# dict of all cetagory infos
CATEGORIES = [
    {
        "supercategory": "H",
        "id": 1,
        "mask_id": 1,
        "name": "H",
        "file_load": [
            "H",
        ],
        "keypoints": [
            "H-1",
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Hftkopf',
        "latex": "$\mathrm{Hip}$",
    },
    {
        "supercategory": "F",
        "id": 2,
        "mask_id": 2,
        "name": "F",
        "file_load": [
            "F"
        ],
        "keypoints": [
            "F-1", "F-2",
            "F-3", "F-4",
            "F-5",
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Femur',
        "latex": "$\mathrm{Femur}_{\mathrm{condyles}}$",
    },
    {
        "supercategory": "T_low",
        "id": 3,
        "mask_id": 3,
        "name": "T_low",
        "file_load": [
            "T",
        ],
        "keypoints": [
            "T-1", "T-2",
            "T-3", "T-4",
            "T-5", "T-6",
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Tibia_H',
        "latex": "$\mathrm{Tibia}_{\mathrm{low}}$",
    },
    {
        "supercategory": "T_up",
        "id": 4,
        "mask_id": 4,
        "name": "T_up",
        "file_load": [
            "T",
        ],
        "keypoints": [
            "T-7",
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Tibia_V',
        "latex": "$$\mathrm{Tibia}_{\mathrm{up}}$",
    },
    {
        "supercategory": "Fi_low",
        "id": 5,
        "mask_id": 5,
        "name": "Fi_low",
        "file_load": [
            "Fi",
        ],
        "keypoints": [
            "Fi-1", "Fi-2",
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Fibula_H',
        "latex": "$\mathrm{Fibula}_{\mathrm{low}}$",
    },
    {
        "supercategory": "Fi_up",
        "id": 6,
        "mask_id": 6,
        "name": "Fi_up",
        "file_load": [
            "Fi",
        ],
        "keypoints": [
            "Fi-3"
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Fibula_V',
        "latex": "$\mathrm{Fibula}_{\mathrm{up}}$",
    },
    {
        "supercategory": "S",
        "id": 7,
        "mask_id": 7,
        "name": "S",
        "file_load": [
            "S",
        ],
        "keypoints": [
            "S-1", "S-2",
            "S-3", "S-4",
            "S-5",
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Talus',
        "latex": "$\mathrm{Ankle}$",
    },
    {
        "supercategory": "A_up",
        "id": 8,
        "name": "A",
        "file_load": [
            "A",
        ],
        "keypoints": [
            "A-1", "A-2",
            "A-3", "A-4",
        ],
        "keypoints_flip_map": [],
        "latex": "$\mathrm{Femur}_{\mathrm{axis}}$",
    },
    {
        "supercategory": "A_low",
        "id": 9,
        "name": "A",
        "file_load": [
            "A",
        ],
        "keypoints": [
            "A-5", "A-6",
            "A-7", "A-8",
        ],
        "keypoints_flip_map": [],
        "latex": "$\mathrm{Tibia}_{\mathrm{axis}}$",
    },
    {
        "supercategory": "O",
        "id": 10,
        "name": "O",
        "file_load": [
            "O",
        ],
        "keypoints": [
            "O-1", "O-2",
            "O-3", "O-4",
            "O-5",
        ],
        "keypoints_flip_map": [],
        "latex": "$\mathrm{Operation}$",
    },
    {
        "supercategory": "K",
        "id": 11,
        "mask_id": 8,
        "name": "K",
        "file_load": [
            "K",
        ],
        "keypoints": [
            "K-1", "K-2",
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Kugel',
        "latex": "$\mathrm{Sphere}$",
    },
    {
        "supercategory": "F_t",
        "id": 12,
        "mask_id": 1,
        "name": "F_t",
        "file_load": [
            "H", "F"
        ],
        "keypoints": [
            "H-1", "F-6",
        ],
        "keypoints_flip_map": [],
        "full_seg": 'Hftkopf',
        "latex": "$\mathrm{Femur}_{\mathrm{trochanter}}$",
    },
]

CATNAMES = [cat['supercategory'] for cat in CATEGORIES]
