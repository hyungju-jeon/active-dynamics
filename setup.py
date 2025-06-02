"""Setup file for the active-dynamics package."""

from setuptools import setup, find_packages

setup(
    name="actdyn",
    version="0.1.0",
    description="Active Learning for Latent Dynamical System Identification",
    author="Hyungju Jeon",
    author_email="hyungju.jeon@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
)
