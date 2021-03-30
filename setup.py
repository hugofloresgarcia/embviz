from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


# TODO - replace with details of your project
setup(
    name='embedding_viz',
    description='',
    version='0.0.1',
    author='Hugo Flores Garcia',
    author_email='hf01049@georgiasouthern.edu',
    url='https://github.com/hugofloresgarcia/embedding-viz',
    install_requires=['plotly', 'pandas'],
    packages=['embedding_viz'],
    package_data={'embedding_viz': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
