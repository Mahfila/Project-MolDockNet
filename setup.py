from setuptools import setup, find_packages

setup(
    name="moldocknet",
    version="1.0.0",
    author="Nur A Mahfila",
    author_email="mahfila2023@gmail.com",
    description="Hybrid GNN + Fingerprint Deep Learning for Molecular Docking Score Prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mahfil/MolDockNet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "rdkit>=2023.3.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
)
