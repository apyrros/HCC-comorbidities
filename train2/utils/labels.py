from enum import IntEnum

class Gender(IntEnum):
    FEMALE = 0
    MALE = 1
    NA = 2

    def convert(s):
        if s == 'female': return Gender.FEMALE
        elif s == 'male': return Gender.MALE
        else: return Gender.NA

class Condition(IntEnum):
    ABSENT = 0
    PRESENT = 1
    NA = 2

    def convert(s):
        if s == 'ABSENT': return Condition.ABSENT
        elif s == 'PRESENT': return Condition.PRESENT
        else: return Condition.NA
