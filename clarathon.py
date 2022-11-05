import matplotlib
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score

from data_handler import read_data
from plot_drawer import plot_age_vs_efficacy_score_and_gender, plot_ad_vs_es_per_country, plot_domain_vs_es_per_gender, \
    plot_experiment_vs_es_per_domain_in_israel

matplotlib.use('TkAgg')


def calculate_efficacy_score(clara_data):
    clara_data['efficacy_score'] = clara_data['Correct (0|1)'] / clara_data['Response Time']
    return clara_data


def calculate_correlation(clara_data):
    clara_data['gender_binary'] = clara_data['Gender'].map({'M': 0, 'F': 1})
    correlation = clara_data.corr()
    print(correlation)


def calculate_efficacy_score_per_subject(clara_data):
    clara_data['subejct_average_efficacy_score'] = clara_data.groupby('Subject')['efficacy_score'].transform('mean')
    return clara_data


def calculate_subjects_number_per_country(clara_data):
    return clara_data.groupby("Country")['Subject'].nunique()


def calculate_mutual_information_between_two_columns(clara_data, column1, column2):
    return mutual_info_regression(clara_data[column1].to_frame(), clara_data[column2], discrete_features=[False])


def calculate_age_normalization_affects_in_domains(clara_data):
    domains = ['space', 'time', 'person']
    for domain in domains:
        domain_clara = clara_data[clara_data['Domain'] == domain]
        print(domain)
        print(calculate_mutual_information_between_two_columns(domain_clara, 'Age', 'efficacy_score'))
        print(calculate_mutual_information_between_two_columns(domain_clara, 'age_normalized_by_retirement',
                                                               'efficacy_score'))
        print(calculate_mutual_information_between_two_columns(domain_clara, 'age_normalized_by_life_expectancy',
                                                               'efficacy_score'))


if __name__ == '__main__':
    clara_data = read_data()
    # plot_response_time_outliers(clara_data)

    clara_data = calculate_efficacy_score(clara_data)
    clara_data = calculate_efficacy_score_per_subject(clara_data)

    plot_experiment_vs_es_per_domain_in_israel(clara_data)

    # print(calculate_mutual_information_between_two_columns(clara_data, 'Age', 'efficacy_score'))
    # print(calculate_mutual_information_between_two_columns(clara_data, 'age_normalized_by_retirement',
    #                                                        'efficacy_score'))
    # print(calculate_mutual_information_between_two_columns(clara_data, 'age_normalized_by_life_expectancy',
    #                                                        'efficacy_score'))

    # clara_data = clara_data[clara_data["Country"] == "China"]
    # plot_domain_vs_es_per_gender(clara_data)

    # plot_age_vs_efficacy_score_and_gender(clara_data)
    # calculate_correlation(clara_data)
    #
    # plot_efficacy_score_line_of_domains(clara_data)

    #
    # calculate_means_of_boys_and_girls(clara_data)
