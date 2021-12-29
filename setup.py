from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='trajectories',
   version='0.1',
   description='easy generation of latent trajectories for generative ML',
   license="MIT",
   long_description=long_description,
   author='Axel Chemla--Romeu-Santos',
   author_email='axelchemla@yahoo.fr',
   url="http://github.com/domkirke",
   packages=['trajectories'],
   install_requires=['wheel', 'scipy>=1.6.2', 'numpy>=1.19.1'],
)