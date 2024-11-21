from setuptools import setup, find_packages

setup(
    name='genaitor',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'flask',
        'gunicorn',
        'flask-limiter',
        'python-dotenv',
        'requests'
    ],
    include_package_data=True,
    package_data={
        '': ['*.llamafile'],  # Make sure the .llamafile is included in the package
    },
)

