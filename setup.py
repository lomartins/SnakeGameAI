from setuptools import setup, find_packages

# Setup snake_rl env
setup(
      name='snake_rl',
      version='0.0.1',
      packages=find_packages(),

      install_requires=['gym', 'pygame'],
      )
