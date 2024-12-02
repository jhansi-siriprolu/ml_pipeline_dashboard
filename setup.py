from setuptools import setup, find_packages

# Read long description from the README file safely
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ml_pipeline_dashboard",  # Your package name
    version="0.1.15",
    py_modules=["ml_pipeline_project"],
    packages=find_packages(),
    install_requires=[
        'pandas',
        'matplotlib',
        'seaborn',
        'plotly',
        'streamlit',  # Add Streamlit as a requirement
    ],
    entry_points={
        'console_scripts': [
            'launch-ml-pipeline=ml_pipeline.launcher:main',  # Entry point for running Streamlit app
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jhansi Siriprolu',
    author_email='siriprolu2018@gmail.com',
    description='A package for ML Pipeline with Streamlit dashboard',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
