# Import distutils
from setuptools import setup, find_packages

setup(
    name="qordering",
    version="0.1",
    description="Qordering Library",
    long_description_content_type="text/markdown",
    author="Sean P. Engelstad",
    author_email="sengeltad312@gatech.edu",
    install_requires=["numpy", "scipy"],
    packages=find_packages(include=["src*"]),
)
