from setuptools import setup, find_packages

setup(
    name="bioconcord",
    version="0.1.0",
    packages=find_packages(where="Src"),
    package_dir={"": "Src"},
    install_requires=[],
)
