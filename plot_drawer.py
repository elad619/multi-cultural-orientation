import pandas as pd
import plotly.express as px
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_age_vs_efficacy_score_and_gender(clara_data):
    clara_data['binned_age'] = pd.qcut(clara_data['Age'], 10)
    sns.barplot(data=clara_data, x="binned_age", y="efficacy_score", hue="Gender")
    plt.show()


def plot_ad_vs_es_per_country(clara_data):
    sns.barplot(data=clara_data, x="Diagnosis", y="efficacy_score", hue="Country")
    plt.show()


def plot_experiment_vs_es_per_domain_in_israel(clara_data):
    israel_clara = clara_data[clara_data['Country'] == 'Israel']
    sns.barplot(data=israel_clara, x="Experiment", y="efficacy_score", hue="Domain")
    plt.show()


def plot_domain_vs_es_per_gender(clara_data):
    sns.barplot(data=clara_data, x="Domain", y="efficacy_score", hue="Gender")
    plt.show()


def plot_response_time_outliers(clara_data):
    fig = px.box(clara_data, y='Response Time')
    fig.show()


def plot_efficacy_score_distribution(clara_data):
    sns.scatterplot(data=clara_data, x="Age", y="efficacy_score", hue="Gender")
    plt.show()


def plot_efficacy_score_histogram(clara_data):
    sns.histplot(data=clara_data, x="Age", y="efficacy_score", hue="Gender")
    plt.show()


def plot_efficacy_score_line(clara_data):
    sns.lineplot(data=clara_data, x="Age", y="efficacy_score", hue="Gender")
    plt.show()


def plot_efficacy_score_line_of_domains(clara_data):
    domains = ['Space', 'Time', 'Person']
    for domain in domains:
        domain_clara = clara_data[clara_data['Domain'] == domain]
        sns.lineplot(data=domain_clara, x="Age", y="efficacy_score", hue="Gender").set(title=domain)
    plt.show()


def plot_linear_regressors_fit(clara_data):
    sns.lmplot(data=clara_data, x="Age", y="efficacy_score", hue="Gender", order=5)
    plt.show()


def plot_linear_regressors_fit_with_domain(clara_data):
    sns.lmplot(data=clara_data, x="Age", y="efficacy_score", hue="Gender", col="Domain", order=5)
    plt.show()
