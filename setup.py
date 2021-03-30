from setuptools import setup, find_packages

from os import path
curdir = path.abspath(path.dirname(__file__))
with open(path.join(curdir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
  name = 'abel-pytorch',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'ABEL Scheduler',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'Vaibhav Balloli',
  author_email = 'balloli.vb@gmail.com',
  url = 'https://github.com/tourdeml/abel-pytorch',
  keywords = [
    'learning rate',
    'pytorch',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)