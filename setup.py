#
# Copyright 2021-2026 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from skbuild import setup
from setuptools import find_packages


setup(
    name="piquassoboost",
    packages=find_packages(
        exclude=(
            "tests", "tests.*",
            "ctests", "ctests.*",
            "performance_tests", "performance_tests.*",
            "piquasso", "piquasso.*",
        )
    ),
    version='0.3.5',
    url="https://github.com/Budapest-Quantum-Computing-Group/piquasso",  # noqa: E501
    maintainer="The Piquasso team",
    maintainer_email="kolarovszki@inf.elte.hu",
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.4",
        "piquasso==7.2.1; platform_system=='Linux'",
    ],
    tests_require=["pytest"],
    description='The C++ binding for the Piquasso project',
    long_description=open("./README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="test, cmake, extension",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C",
        "Programming Language :: C++"
    ],
    license='Apache Software License',
    scripts=[
        "cmake/check_AVX.cmake",
        "cmake/FindBLASEXT.cmake",
        "cmake/FindCBLAS.cmake",
    ],
)
