from setuptools import setup, find_packages

setup(
    name="jax_rnafold",
    version="2.0.0-beta",  # Beta version to signal it's an early release
    description="A package for differentiable RNA folding and design.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ryan Krueger",
    author_email="rkruegs123@gmail.com",
    url="https://github.com/rkruegs123/jax_rnafold",  # Update with your repository URL
    license="Your University License",  # Update with your specific license
    package_dir={"": "src"},  # Specifies the root directory for packages
    packages=find_packages(where="src", exclude=["tests"]),  # Finds all packages under src
    include_package_data=True,  # Includes files specified in MANIFEST.in
    install_requires=[
        "numpy",
        "tqdm",
        "jax",
        "optax",
        "pandas",
        "viennarna",
        "biopython",
        "matplotlib",
        "flax"
    ],
    extras_require={
        "docs": ["sphinx", "sphinx-autodoc-typehints", "sphinx-book-theme", "sphinx-copybutton"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        # Removed OSI Approval since it's not applicable
        "License :: Other/Proprietary License",  # Generic non-OSI classifier
        "Operating System :: OS Independent",
    ],
)
