import setuptools

if __name__ == "__main__":
    setuptools.setup(name="recommender_pkg",
                     version="0.0.1",
                     author="Mian Uddin",
                     description="A small recommender package",
                     packages=["recommender_pkg"],
                     install_requires=["numpy",
                                       "pandas",
                                       "seaborn",
                                       "sklearn",
                                       "tensorflow",
                                       "tqdm"])
