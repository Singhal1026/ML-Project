from setuptools import setup, find_packages
from typing import List

PROJECT_NAME = "Machine Learning Project"
VERSION = "0.0.1"
DESCRIPTION = "ML Project Project Modular Coding"
AUTHOR = "Yash Singhal"
EMAIL = "example1234@gmail.com"

HYPHEN_E_DOT = "-e ."


def get_requirements() -> List[str]:
    requirement_list = []
    with open("requirements.txt", "r") as f:
        requirement_list = [line.strip() for line in f.read().split("\n") if line.strip()]
    
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)

    return requirement_list


setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    packages=find_packages(),
    install_requires=get_requirements(),
)
