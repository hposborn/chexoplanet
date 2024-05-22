from setuptools import setup
import setuptools

with open("ReadMe.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='chexoplanet',
    version='0.0.1',
    description='A package for modelling CHEOPS data with exoplanet.',
    url='https://github.com/hposborn/chexoplanet',
    author='Hugh P. Osborn',
    author_email='hugh.osborn@unibe.ch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='BSD 2-clause',
    project_urls={
        "Bug Tracker": "https://github.com/hposborn/MonoTools/issues",
    },
    packages=setuptools.find_packages(),
    install_requires=['matplotlib',
                      'numpy<1.22',
                      'pandas',
                      'scipy>1.7.3',
                      'astropy',
                      'astroquery',
                      'arviz',
                      'pymc3<=3.11.5',
					  'pymc3_ext',
                      'exoplanet==0.5.3',
                      'celerite2',
                      'requests',
                      'urllib3',
                      'lxml',
                      'httplib2',
                      'h5py',
                      'corner',
                      'transitleastsquares',
                      'seaborn',
                      'patsy',
                      'tess-point'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
    ],
)
