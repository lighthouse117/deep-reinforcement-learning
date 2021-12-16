from enum import IntEnum


class StockRemaining(IntEnum):
    MANY = 0
    FEW = 1
    NONE = 2


class StockChange(IntEnum):
    NONE = 0
    SLIGHT = 1
    GREAT = 2


class Satisfaction(IntEnum):
    NOT = 0
    SOMEWHAT = 1
    PERFECT = 2
    pass
