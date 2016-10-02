from distutils.core import setup
import subprocess
from setuptools import find_packages
import sys

lib_filename = None
if sys.platform == 'darwin':
    lib_filename = 'libtensorflow-xlogx.dylib'
else:
    lib_filename = 'libtensorflow-xlogx.so'

subprocess.call(["cmake","."])
subprocess.call(["make"])
subprocess.call(["cp",lib_filename,"python-src/"+lib_filename])

SRC_DIR = './python-src/'

setup(
    name='tensorflow-xlogx',
    version='0.2dev',
    package_dir={'tensorflow_xlogx': SRC_DIR},
    packages=['tensorflow_xlogx'],
    license='All rights reserved.',
    long_description=open('README.md').read(),
    zip_safe=False,
    package_data={'tensorflow_xlogx': [lib_filename]},
    include_package_data=True
)
