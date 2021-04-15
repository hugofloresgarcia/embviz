from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


# TODO - replace with details of your project
setup(
    name='embviz',
    description='',
    version='0.0.1',
    author='Hugo Flores Garcia',
    author_email='hf01049@georgiasouthern.edu',
    url='https://github.com/hugofloresgarcia/embviz',
    install_requires=['plotly', 'pandas', 'sklearn', 'umap'],
    packages=['embviz'],
    package_data={'embviz': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
