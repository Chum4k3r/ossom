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
    'version': '0.1.0a',
    'description': 'Audio IO tools for real time data visualization',
    'url': 'http://github.com/Chum4k3r/ossom',
    'author': 'João Vitor Gutkoski Paes',
    'author_email': 'joao.paes@eac.ufsm.br',
    'license': 'MIT',
    'install_requires': ['numpy', 'scipy', 'soundcard', 'numba'],
    'packages': ['ossom', 'ossom.utils'],
    'package_dir': {'utils': 'ossom'},
    'package_data': {'examples': ['examples/*.py']}
}

setup(**settings)

