#!/usr/bin/env python3
"""
Setup script for DeepTrix-Z
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
    name="deeptrix-z",
    version="0.1.0",
    author="DeepTrix-Z Team",
    author_email="contact@deeptrix-z.com",
    description="DeepTrix-Z: The Ultimate Modern Tetris Bot",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/deeptrix-z",  # Replace with your repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Puzzle Games",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "deeptrix-z=main:main",
        ],
    },
    keywords=[
        "tetris",
        "ai",
        "reinforcement-learning",
        "mcts",
        "neural-network",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/your-username/deeptrix-z/issues",
        "Source Code": "https://github.com/your-username/deeptrix-z",
    },
) 