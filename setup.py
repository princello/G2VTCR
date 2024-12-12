from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, "r") as file:
        return file.read().splitlines()
setup(
    name="g2vtcr",
    version="1.0.0",
    author="Shen Lab at Columbia University",
    author_email="zw2595l@cumc.columbia.edu",
    description="Graph-based Representation and Embedding of Antigen and TCR for Enhanced Recognition Analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/princello/g2vtcr",    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": parse_requirements("requirements-dev.txt"),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
