from setuptools import setup, find_packages

setup(
    name='mfid',
    version='0.1.0',
    author= 'Quentin Bacquelé',
    author_email= 'quentin.bacquele@etu.unistra.fr',
    url= 'https://github.com/quentinbacquele/mfid',
    packages=find_packages(),
    install_requires=[
        'PyQt5',
        'ultralytics',
        'numpy',
        'opencv-python',
        'Pillow',
        'PyYAML',
        'requests',
        'scipy',
        'torch',
        'torchvision',
        'tqdm',
        'pyqt5',
        'pandas',
        'seaborn'
    ],
    entry_points={
        'console_scripts': [
            'mfid=mfid.cli:main',
        ],
    },
)
