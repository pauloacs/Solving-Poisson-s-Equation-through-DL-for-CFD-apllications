from setuptools import setup, find_packages

setup(
    name='pressureSM_deltas',
    version='0.1',
    packages=find_packages(where='source'),
    package_dir={'': 'source'},
    entry_points={
        'console_scripts': [
            'train_script = pressureSM_deltas.entry_point:train_entry_point',
            'evaluation_script = pressureSM_deltas.entry_point:eval_entry_point',
        ]
    },
    install_requires=[
        'numpy',
        'tensorflow'
    ],
)