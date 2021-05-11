from setuptools import find_packages, setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="code_source",
    packages=find_packages(),
    version="0.1.0",
    description="HW1 project",
    author="Olga Novokreschenova",
    license="MIT",
    install_requires=[
        "click==7.1.2",
        "coverage==5.5",
        "setuptools==56.2.0",
        "python-dotenv>=0.5.1",
        "pytest==6.2.3",
        "pandas==1.2.4",
        "scikit-learn==0.24.1",
        "dataclasses>=0.6",
        "pyyaml==5.3",
        "marshmallow-dataclass==8.3.0",
        "numpy==1.20.2"
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'

)
