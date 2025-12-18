import setuptools
__MAJOR__ = 0
__MINOR__ = 1
__MICRO__ = 0
__RELEASE__ = ''  # a #b  # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers --> alpha beta, ...
__VERSION__ = ''.join([str(__MAJOR__), '.', str(__MINOR__), '.', str(__MICRO__)])
__DESCRIPTION__ = """A comprehensive Python pipeline for detecting, quantifying, and analyzing FISH (Fluorescence In Situ Hybridization) spots in 3D microscopy images using deep learning and chromatic aberration correction."""
__URL__ = 'https://github.com/baigouy/FISH-Dist'
__EMAIL__ = 'benoit.aigouy@gmail.com'
__AUTHOR__ = 'Benoit Aigouy'

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='fishdist', # shall I do it like that or not --> try on testpy first
    version=__VERSION__,
    author=__AUTHOR__,
    author_email=__EMAIL__,
    description=__DESCRIPTION__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=__URL__,
    package_data={'': ['*.md']},
    license='BSD',
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        "czifile",
        "Markdown",
        "matplotlib>=3.5.2",
        "numpy<1.26.4", # <1.26.4 for compat with javabridge
        "Pillow>=8.1.2",
        "PyQt6",
        "read-lif",
        "scikit-image>=0.19.3",
        "scipy>=1.7.3",
        "scikit-learn>=1.0.2",
        "tifffile>=2021.11.2",
        "tqdm",
        "natsort",
        "numexpr",
        "urllib3",
        "qtawesome",
        "pandas",
        "numba",
        "elasticdeform",
        "roifile",
        "prettytable",
        "pyperclip",
        "QtPy>=2.1.0",
        "Deprecated",
        "Requests",
        # "python-bioformats",
        # "python-javabridge",
        "pyautogui",
        "imagecodecs",
        "psutil",
        "batoolset",
        # "pyfigures",
        # "zarr",
    ],
    extras_require={'all': [
        "python-javabridge",
        "python-bioformats",
    ],
    },
    python_requires='>=3.10, <3.11' # from 04/05/23 colab is using python 3.10.11
)