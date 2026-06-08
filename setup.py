from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    description = f.read()

setup(
    name='keras-nac',
    version='0.0.4',
    description='A Tensorflow/Keras implementation of the Neuronal Attention Circuit (NAC) and variants.',
    author='Waleed Razzaq',
    packages=find_packages(),
    install_requires=[
        'ncps==1.0.1',
    ],
    long_description=description,
    long_description_content_type='text/markdown',
)

