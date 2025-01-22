from setuptools import setup, find_packages

setup(
    name='norma',
    version='0.1.0',
    python_requires='>=3.9',
    packages=find_packages(where='./src/', include=['*']),
    package_dir={'': 'src'},
    zip_safe=False,
    install_requires=[],
    platforms='any'
)
