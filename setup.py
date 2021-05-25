import setuptools
from os import path

if __name__ == "__main__":
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    setuptools.setup(name="recommender-pkg",
                     version="0.0.4",
                     author="Mian Uddin",
                     author_email="mianuddin@gmail.com",
                     description="A small recommender package",
                     long_description=long_description,
                     long_description_content_type="text/markdown",
                     url="https://github.com/mianuddin/csc492_recommender_pkg",
                     packages=["recpkg"],
                     install_requires=["numpy==1.19.5",
                                       "pandas==1.2.4",
                                       "seaborn==0.11.1",
                                       "scikit-learn==0.24.1",
                                       "tensorflow==2.5.0rc1",
                                       "tqdm==4.60.0"])
