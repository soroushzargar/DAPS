from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...

setup(
    name="gnn_cp",
    version="0.0.1",
    author=["Soroush H. Zargarbashi"],
    author_email=["sayed.haj-zargarbashi@cispa.de"],
    description=("Conformal Prediction Sets for Graph Neural Networks"),
    license="BSD",
    keywords="base",
    packages=find_packages(),
    long_description='',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)