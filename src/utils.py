import pandas as pd
import numpy as np

def categorize_age(age):
    """Categorizes age into standard age groups."""
    if 0 <= age <= 9:
        return 'Child'
    elif 10 <= age <= 19:
        return 'Adolescent'
    elif 20 <= age <= 35:
        return 'Young Adult'
    elif 36 <= age <= 45:
        return 'Middle-Aged Adult'
    elif 46 <= age <= 60:
        return 'Senior Adult'
    elif age > 60:
        return 'Senior Citizen'
    else:
        return 'Unknown'


def BMI(height, weight):
    """Computes Body Mass Index (BMI) and maps to WHO categories."""
    m = height / 100
    val = (weight / m**2)
    val = round(val, 1)
    if val < 18.5:
        return 'Underweight'
    elif 18.5 <= val <= 24.9:
        return 'Normal weight'
    elif 25 <= val <= 29.9:
        return 'Overweight'
    elif val >= 30:
        return 'Obese'
    else:
        return 'unknown'


def eye(x, y):
    """Computes the rounded average of left and right eyesight or hearing values."""
    return ((x + y) / 2).round(1)
