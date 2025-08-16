#!/usr/bin/env python3
"""
Setup script for Performance ML System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="perfml",
    version="1.0.0",
    author="Performance ML Team",
    author_email="team@perfml.com",
    description="A comprehensive Machine Learning-driven performance testing and optimization system",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/perfml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "perfml=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    keywords="performance testing, machine learning, load testing, optimization, automation",
    project_urls={
        "Bug Reports": "https://github.com/your-org/perfml/issues",
        "Source": "https://github.com/your-org/perfml",
        "Documentation": "https://perfml.readthedocs.io/",
    },
)
