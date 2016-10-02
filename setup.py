from distutils.core import setup
import subprocess
from setuptools import find_packages

subprocess.call(["cmake","."])
subprocess.call(["make"])

SRC_DIR = './python-src/'

setup(
    name='tensorflow-xlogx',
    version='0.2dev',
    package_dir={'tensorflow_xlogx': SRC_DIR},
    packages=['tensorflow_xlogx'],
    license='All rights reserved.',
    long_description=open('README.md').read(),
    zip_safe=False,
    package_data={'tensorflow_xlogx': ['libtensorflow-xlogx.so']},
    include_package_data=True
)