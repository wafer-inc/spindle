from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spindle",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A library for training and deploying Sparse Autoencoders",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/spindle",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
    ],
    extras_require={
        "transformers": ["transformers>=4.0.0", "sentence-transformers>=2.0.0"],
        "database": ["sqlite3"],
        "server": ["fastapi>=0.68.0", "uvicorn>=0.15.0", "pydantic>=1.8.0"],
    }
)