from skbuild import setup
from setuptools import find_packages


setup(
    name="cpiquasso",
    packages=find_packages(),
    version='0.1',
    url="https://gitlab.inf.elte.hu/wigner-rcp-quantum-computing-and-information-group/cpiquasso",  # noqa: E501
    maintainer="The Piquasso team",
    maintainer_email="kolarovszki@inf.elte.hu",
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.4",
        "quantum-blackbird>=0.2.4",
        "BoSS-Tomev>=0.0.6",
        "tbb"
    ],
    tests_require=["pytest"],
    description='The C++ binding for the Piquasso project',
    long_description=open("./README.md", 'r').read(),
    long_description_content_type="text/markdown",
    keywords="test, cmake, extension",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: "
        "Apache License 2.0.",
        "Natural Language :: English",
        "Programming Language :: C",
        "Programming Language :: C++"
    ],
    license='Apache License 2.0.',
)
