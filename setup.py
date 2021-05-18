import pathlib
from setuptools import setup, find_packages, Extension

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="poeem",
    version="0.9",
    description="A library for jointly training embedding retrieval model and product quantization based index",
    long_description=README,
    long_description_content_type="text/markdown",
    #url="https://github.com/realpython/reader",
    author="JD.com",
    author_email="wenyun.yang@jd.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    package_dir={"poeem": "src", "poeem.python": "src/python", "poeem.ops.bin": "src/ops/bin", "poeem.ops.python": "src/ops/python"},
    packages=["poeem", "poeem.ops.python", "poeem.ops.bin", "poeem.python"],
    include_package_data=True,
    install_requires=["tensorflow-gpu==1.15"],
)
