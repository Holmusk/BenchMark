from setuptools import setup, find_packages

setup(
    name='benchtool',
    version='1.0.0',
    description='A lightweight tool to benchmark the performance of NLP models based on medical text datasets.',
    author='Holmusk',
    author_email='varun.c@holmusk.com',
    packages=find_packages(include=['utils', 'utils.*']),
    py_modules=['main', 'benchmark'],
    install_requires=[
        r.strip() for r in open('requirements.txt').readlines() if r.strip() and not r.startswith('#')
    ],
    entry_points={
        'console_scripts': [
            'benchtool = main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['config.yml']
    },
    python_requires='>=3.7',
) 