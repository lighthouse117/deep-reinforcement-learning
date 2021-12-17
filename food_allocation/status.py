from enum import IntEnum


class StockRemaining(IntEnum):
    FULL = 0
    MANY = 1
    FEW = 2
    NONE = 3


class StockChange(IntEnum):
    NONE = 0
    SLIGHT = 1
    SOMEWHAT = 2
    GREAT = 3


class Satisfaction(IntEnum):
    NOT = 0
    SLIGHT = 1
    SOMEWHAT = 2
    PERFECT = 3
    pass
