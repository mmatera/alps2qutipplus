from distutils.core import setup

setup(
    name='Alps2Qutip',
    version='1.0',
    packages=['alpsqutip'],
    url='http://www.yourwebsite.com',
    license='LICENSE.txt',
    author='Juan Mauricio Matera',
    author_email='matera@fisica.unlp.edu.ar',
    description='Your project description',
    long_description=open('README.md').read(),
    install_requires=[
        'matplotlib',
        'qutip',
    ],
)
