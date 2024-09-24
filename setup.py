from setuptools import find_packages, setup

setup(
    name='disclinations',
    version='2024.0.1',
    package_dir={'': 'src'},  # The root package is under the 'src' directory
    packages=find_packages('src'),  # Find packages under the 'src' directory
    include_package_data=True,  # Ensure non-Python files are included
    package_data={
        'disclinations': ['data/*.yml', 'models/*.yml', 'test/parameters.yml'],
    },
)
