from setuptools import setup, find_packages

setup(
    name='mfid',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'PyQt5',
        # any other dependencies your project needs
    ],
    entry_points={
        'console_scripts': [
            'mfid=mfid.cli:main',
        ],
    },
)
