#!/usr/bin/env python

from setuptools import setup

setup(
    name='metaforce',
    version='0.0.0',
    description='TODO',
    url='TODO',
    author='PENG Zhenghao, LI Yunxiang',
    author_email='pengzh@ie.cuhk.edu.hk, TODO',
    packages=['metaforce'],
    install_requires=[
        "metaworld@git+https://github.com/rlworkgroup/metaworld.git"
        "@a3e80c2439aa96ff221d6226bcf7ab8b49689898",
        "yapf==0.27.0", "ray==0.8.7", "pandas", "tensorboardX", "tabulate",
        "gym[box2d]", "ray[rllib]"
    ]
)
