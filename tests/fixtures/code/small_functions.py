"""Small top-level functions plus module-level code (R1, R2, R5 fixture)."""
import math
import os

GREETING = "hello"
MAX_RETRIES = 3


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    if b == 0:
        raise ZeroDivisionError("b must not be zero")
    return a / b


def hypot(a, b):
    return math.sqrt(a * a + b * b)


def env_name():
    return os.environ.get("NAME", GREETING)
