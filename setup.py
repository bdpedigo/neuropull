import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ["pandas", "requests", "networkx"]

setuptools.setup(
    name="neuropull",
    version="0.0.1",
    author="Benjamin Pedigo",
    author_email="bpedigo@jhu.com",
    description="Pulls connectomics datasets as graphs with metadata",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bdpedigo/neuropull",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
