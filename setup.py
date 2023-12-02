import os
from typing import List
from setuptools import find_packages, setup

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path) -> List[str]:
    """A function to get requirements for installation.
    
    Args: 
        file_path: Define file location of object.

    Returns: 
        List of python libraries
    """
    requirements = []

    with open(file_path, 'r') as f:
        requirements = f.readlines()
        requirements = [requirement.replace('\n', '') for requirement in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        
    return requirements


# For quick project installation
setup(
    name='mlproject',
    version='0.0.1',
    author='Dennis',
    author_email='dkb7826@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

