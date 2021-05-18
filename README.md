# Recommender Package
Implementations of explicit and implicit recommenders with Numpy and Keras.

```sh
$ pip install recommender-pkg
```

**Author:** Mian Uddin<br>
**Advisor:** Dr. Paul Anderson<br>
**Course(s):** CSC 491, CSC 492<br>
**School:** Cal Poly, San Luis Obispo

## Usage
```python3
import recpkg
```

See the [documentation](https://csc492-recommender-pkg.readthedocs.io/) for
more.

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