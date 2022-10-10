import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="automatic_vehicular_control", # Replace with your own username
    version="0.0.1",
    author="Zhongxia \"Zee\" Yan",
    author_email="zxyan@mit.edu",
    description="Code for Unified Automatic Control of Vehicular Systems With Reinforcement Learning (IEEE T-ASE 2022)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mit-wu-lab/automatic_vehicular_control",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
