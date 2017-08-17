# -*- coding: utf-8 -*-

# setuptools facilitates packaging Python projects by enhancing the Python standard library
from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

setup(
    name='STSM-Prediction-Service',
    version='0.1.0',
    description='STSM prediction',
    long_description=readme,
    author='Rotem Jordan',
    author_email='rotemjordan@gmail.com',
    packages=find_packages(), # TODO in case we will want to exclude packages(exclude=('tests', 'docs')),
    # todo needed? data_files=[('model', ['model/feature_names.csv', 'model/pipeline.pkl', 'model/risk_model.h5'])]
)
