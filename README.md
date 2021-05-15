# Recommender Package

CSC 491<br>
CSC 492

Mian Uddin

## Build
Use the following command to build a wheel of this package.

```python
python3 setup.py bdist_wheel --universal
```

## Test
Use the following command to run unit tests.
```python
python3 -m unittest tests
```

## Document
Use the following command to build the documentation.
```sh
sphinx-apidoc -f -o docs/source recommender_pkg/
(cd docs/ && make html)
```