from distutils.core import setup


setup(
    name='streaming',
    version='2.0.0',  # Format.Feature.Bugfix
    description='Streaming datasets',
    author='MosaicML',
    author_email='team@mosaicml.com',
    url='https://github.com/mosaicml/streaming/',
    packages=['streaming'],
    install_requires=open('requirements.txt').readlines(),
)
