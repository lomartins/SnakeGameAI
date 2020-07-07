from setuptools import setup, find_packages

setup(
      name='snake_rl',
      version='0.0.1',
      packages=find_packages(),

      install_requires=['gym',
                        'keras-rl',
                        'numpy==1.16.4',
                        'tensorflow==1.13.1',
                        'keras==2.2.4',
                        'pygame'
                        ],
      package_data={'binary_rl_forex': ['datasets/data/*']}
      )
