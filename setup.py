from setuptools import setup
import setuptools

with open("ReadMe.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='chexoplanet',
    version='0.1.0',
    description='A package for modelling CHEOPS data with exoplanet.',
    url='https://github.com/hposborn/chexoplanet',
    author='Hugh P. Osborn',
    author_email='hugh.osborn@unibe.ch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD 2-clause',
    project_urls={
        "Bug Tracker": "https://github.com/hposborn/chexoplanet/issues",
    },
    packages=setuptools.find_packages(),
    install_requires=['exoplanet[pymc]',
                      'pymc-ext',
                      'dace_query',
                      'celerite2',
                      'pymc',
                      'matplotlib',
                      'numpy',
                      'pandas',
                      'scipy<1.13',
                      'astropy',
                      'astroquery',
                      'arviz',
                      'aesara_theano_fallback',
                      'requests',
                      'urllib3',
                      'lxml',
                      'httplib2',
                      'h5py',
                      'corner',
                      'transitleastsquares',
                      'seaborn',
                      'patsy',
                      'tess-point',
                      'everest-pipeline',
                      'bs4',
                      'lightkurve',
                      'tess-point',
                      'iteround',
                      'bokeh'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
    ],
)
