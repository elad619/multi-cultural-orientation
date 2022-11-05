import pandas as pd
import numpy as np


def read_data():
    clara_data = pd.read_excel('data/clara_data.xlsx', sheet_name='data')
    clara_data = handle_problematic_response_times(clara_data)
    clara_data = drop_problematic_correct_labels(clara_data)
    clara_data = drop_problematic_younger_age(clara_data)
    clara_data = drop_controls(clara_data)
    clara_data = standardize_domain_column(clara_data)
    clara_data = standardize_diagnosis_column(clara_data)
    clara_data = add_country_column(clara_data)
    clara_data = add_age_normalized_by_retirement_column(clara_data)
    clara_data = add_age_normalized_by_life_expectancy_column(clara_data)

    return clara_data


def handle_problematic_response_times(clara_data):
    clara_data = clara_data[clara_data['Response Time'] > 0]
    clara_data['Response Time'] = clara_data['Response Time'].clip(upper=5,
                                                                   lower=0.2)
    return clara_data


def drop_problematic_correct_labels(clara_data):
    clara_data = clara_data[clara_data['Correct (0|1)'].isin([0, 1])]
    return clara_data


def drop_controls(clara_data):
    clara_data = clara_data[clara_data['Domain'] != 'Control']
    clara_data = clara_data[clara_data['Domain'] != 'control']
    return clara_data


def drop_problematic_younger_age(clara_data):
    clara_data = clara_data[clara_data['Age'] > 4]
    return clara_data


def transform_place_to_space(clara_data):
    return clara_data


def standardize_domain_column(clara_data):
    clara_data['Domain'] = clara_data['Domain'].str.lower()
    clara_data['Domain'] = clara_data['Domain'].str.replace('place', 'space')
    return clara_data


def standardize_diagnosis_column(clara_data):
    clara_data['Diagnosis'] = np.where(clara_data['Diagnosis'] == 'OC', 'HOC', clara_data['Diagnosis'])
    clara_data['Diagnosis'] = np.where(clara_data['Diagnosis'] == 'HYC', 'YC', clara_data['Diagnosis'])
    clara_data['Diagnosis'] = np.where(clara_data['Diagnosis'] == 'HC', 'HOC', clara_data['Diagnosis'])
    return clara_data


def add_country_column(clara_data):
    experiment_to_country_mapping = {
        1: 'Israel',
        2: 'Israel',
        3: 'Israel',
        4: 'Switzerland',
        5: 'Israel',
        6: 'Brazil',
        7: 'USA',
        8: 'Israel',
        9: 'China'
    }
    clara_data['Country'] = clara_data['Experiment'].map(experiment_to_country_mapping)
    return clara_data


retirement_ages_mapping = {
    'Israel': {'M': 67, 'F': 62},
    'Switzerland': {'M': 65, 'F': 64},
    'Brazil': {'M': 62, 'F': 57},
    'USA': {'M': 66, 'F': 66},
    'China': {'M': 60, 'F': 55}
}


def calculate_age_normalized_by_retirement_of_row(data_row):
    return data_row['Age'] / retirement_ages_mapping[data_row['Country']][data_row['Gender']]


def add_age_normalized_by_retirement_column(clara_data):
    clara_data['age_normalized_by_retirement'] = clara_data.apply(calculate_age_normalized_by_retirement_of_row, axis=1)
    return clara_data


life_expectancy_mapping = {
    'Israel': {'M': 81, 'F': 84.3},
    'Switzerland': {'M': 81.6, 'F': 85.4},
    'Brazil': {'M': 76.3, 'F': 81.3},
    'USA': {'M': 71.9, 'F': 79.3},
    'China': {'M': 74.5, 'F': 79}
}


def calculate_age_normalized_by_life_expectancy(data_row):
    return data_row['Age'] / life_expectancy_mapping[data_row['Country']][data_row['Gender']]


def add_age_normalized_by_life_expectancy_column(clara_data):
    clara_data['age_normalized_by_life_expectancy'] = clara_data.apply(calculate_age_normalized_by_life_expectancy,
                                                                       axis=1)
    return clara_data
