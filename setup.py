from setuptools import setup, find_packages

setup(
    name='sierra',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    author='Jan Haslik',
    author_email='jan@haslik.at',
    description='A Python library for financial and quantitative finance calculations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Blue-SeaBird/Sierra',
)
