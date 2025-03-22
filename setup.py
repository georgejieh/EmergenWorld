"""Setup script for EmergenWorld package."""

from setuptools import setup, find_packages

setup(
    name="emergenworld",
    version="0.1.0",
    description="A fantasy world simulator where AI species evolve naturally",
    author="George Jieh",
    author_email="george.jieh@gmail.com",
    url="https://github.com/georgejieh/EmergenWorld",
    packages=find_packages(include=['src', 'src.*']),
    package_data={
        "": ["*.md", "*.txt"],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "opensimplex",
        "ephem",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)