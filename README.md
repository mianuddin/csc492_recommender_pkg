# Recommender Package
Implementations of explicit and implicit recommenders with Numpy and Keras.

```sh
$ pip install recommender-pkg
```

**Author:** Mian Uddin

**Advisor:** Dr. Paul Anderson

**Course(s):** CSC 491, CSC 492

**School:** Cal Poly, San Luis Obispo

## Build
Use the following command to build a wheel of this package.

```sh
$ python3 -m build
```

## Test
Use the following command to run unit tests.
```sh
$ python3 -m unittest tests
```

## Document
Use the following command to build the documentation.
```sh
$ sphinx-apidoc -f -o docs/source recommender_pkg/
$ (cd docs/ && make html)
```

## Release
Use the following command to distribute to PyPi.
```sh
$ python3 -m twine upload dist/*
```