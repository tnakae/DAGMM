from distutils.core import setup

setup(
    name='DAGMM',
    packages=['dagmm',],
    license='MIT License',
    author='Toshihiro NAKAE',
    description='UNOFFICIAL implementation of the paper Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection [Bo Zong et al (2018)]',
    long_description=open('README.md').read(),
)