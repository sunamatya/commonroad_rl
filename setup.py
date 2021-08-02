#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""commonroad_rl setup file."""

import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    print("Please install or upgrade setuptools or pip to continue")
    sys.exit(1)

setup(
    name="commonroad-rl",
    version="2020.2",
    packages=find_packages(
        exclude=["tests", "planning", "utils_run"]
    ),
    package_data={"": ["*.xml", "*.pickle"]},
    description="Tools for applying reinforcement learning on commonroad scenarios.",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    test_suite="commonroad_rl.tests",
    keywords="autonomous automated vehicles driving motion planning".split(),
    url="https://commonroad.in.tum.de/",
    install_requires=[
        "networkx",
        "triangle",
        "commonroad-io>=2021.1",
    ],
    extras_require={
        "utils_run": ["optuna", "PyYAML"],
        "tests": ["pytest"],
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Operating System :: POSIX :: Linux",
    ],
)
