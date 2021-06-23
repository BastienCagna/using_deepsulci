#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages
from distutils.core import setup

setup(
    name="using_deepsulci",
    version='nc',
    packages=find_packages(),
    description="Custom script to work arround deepsulci",
    install_requires=["numpy", "capsul", "soma"],
)