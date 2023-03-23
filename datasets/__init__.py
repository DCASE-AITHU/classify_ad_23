CLASS_MAP20 = {
    'fan': 0,
    'pump': 1,
    'slider': 2,
    'ToyCar': 3,
    'ToyConveyor': 4,
    'valve': 5
}
INVERSE_CLASS_MAP20 = {
    0: 'fan',
    1: 'pump',
    2: 'slider',
    3: 'ToyCar',
    4: 'ToyConveyor',
    5: 'valve'
}
TRAINING_SECTION_MAP20 = {
    0: [0, 2, 4, 6],
    1: [0, 2, 4, 6],
    2: [0, 2, 4, 6],
    3: [1, 2, 3, 4],
    4: [1, 2, 3],
    5: [0, 2, 4, 6]
}
EVALUATION_SECTION_MAP20 = {
    0: [1, 3, 5],
    1: [1, 3, 5],
    2: [1, 3, 5],
    3: [5, 6, 7],
    4: [4, 5, 6],
    5: [1, 3, 5]
}
ALL_SECTION_MAP20 = {
    0: [0, 1, 2, 3, 4, 5, 6],
    1: [0, 1, 2, 3, 4, 5, 6],
    2: [0, 1, 2, 3, 4, 5, 6],
    3: [1, 2, 3, 4, 5, 6, 7],
    4: [1, 2, 3, 4, 5, 6],
    5: [0, 1, 2, 3, 4, 5, 6]
}

# bearing  fan  gearbox  slider  ToyCar  ToyTrain  valve
CLASS_MAP23 = {
    'bearing': 0,
    'fan': 1,
    'gearbox': 2,
    'slider': 3,
    'ToyCar': 4,
    'ToyTrain': 5,
    'valve': 6,
}
INVERSE_CLASS_MAP23 = {
    0: 'bearing',
    1: 'fan',
    2: 'gearbox',
    3: 'slider',
    4: 'ToyCar',
    5: 'ToyTrain',
    6: 'valve',
}
DEV_TYPES23 = [
    'bearing',
    'fan',
    'gearbox',
    'slider',
    'ToyCar',
    'ToyTrain',
    'valve',
]
EVAL_TYPES23 = []
ALL_TYPES23 = DEV_TYPES23 + EVAL_TYPES23
