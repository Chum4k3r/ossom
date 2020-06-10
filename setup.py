# -*- coding: utf-8 -*-
"""
Ossom setup file
=================

@Author:
- João Vitor Gutkoski Paes, joao.paes@eac.ufsm.br

"""

from setuptools import setup

settings = {
    'name': 'OsSom',
    'version': '0.1.0',
    'description': 'Audio IO tools for data visualization',
    'url': 'http://github.com/Chum4k3r/ossom',
    'author': 'João Vitor Gutkoski Paes',
    'author_email': 'joao.paes@eac.ufsm.br',
    'license': 'MIT',
    'install_requires': ['numpy', 'scipy', 'sounddevice', 'numba'],
    'packages': ['ossom', 'ossom.audio', 'ossom.utils'],
    'package_dir': {'audio': 'ossom', 'utils': 'ossom'},
    'package_data': {'tests': ['tests/*.py']}
}

setup(**settings)
