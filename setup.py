from setuptools import setup, find_packages

setup(
    name='benchtool',
    version='0.1.0',
    description='A lightweight tool to benchmark the performance of NLP models based on medical text datasets.',
    author='Holmusk',
    author_email='varun.c@holmusk.com',
    packages=find_packages(include=['utils', 'utils.*']),
    py_modules=['main', 'benchmark'],
    install_requires=[
        'torch==2.7.1',
        'transformers==4.48.2',
        'optimum==1.26.1',
        'onnxruntime==1.22.0',
        'onnx==1.18.0',
        'pandas==2.2.3',
        'tqdm==4.67.1',
        'PyYAML==6.0.2',
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