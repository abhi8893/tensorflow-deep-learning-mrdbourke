from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()


setup(
    author='Abhishek Bhatia',
    author_email='bhatiaabhishek8893@gmail.com',
    name='src',
    version='0.0.1',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=requirements

)