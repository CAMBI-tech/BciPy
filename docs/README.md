# BciPy Documentation

This directory contains the documentation for BciPy that is built using Sphinx.

## Building the API Documentation

To build the API documentation, run the following command from the root directory:

```bash
make build-api-docs
```

## Viewing the Documentation

To view the documentation, open the `build/html/index.html` file in your browser.

## Serving the Documentation

To serve the documentation, run the following command from the root directory:

```bash
make serve-docs
```

This will serve the documentation at `http://localhost:8000`.

## Contributing to the Documentation

To contribute to the documentation, edit the files in the `docs` directory. If changes were made to the API docs, run the `make build-api-docs`. After this is done, change directory into the docs folder and generate the html files.

```bash
cd docs
make clean
make html
```

This will generate the html files and they will be located in the `_build/html` directory. These can be viewed by running `make serve-docs` or by opening the `index.html` file in your browser.
