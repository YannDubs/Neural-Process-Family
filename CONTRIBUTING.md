# Contributing

## Online Book
We are using [jupyter book](https://jupyterbook.org/intro.html) for the website. The book is located in `jupyter/`

**writing**
- the entry point for the writing the book is `jupyter/_config.yml` which gives the table of content, i.e. which files go where. From there you can easily find the important files (`*.ipynb` or `*.md`) to change, or add new ones your own.

**Publish and view**
- run `jupyter-book build jupyter/`
- view locally by opening `file://[path]/jupyter/_build/html/intro.html` in your browser (replace `[path]`)
- **ONLY TO PUT IT PUBLIC** upload to github pages `ghp-import -n -p -f jupyter/_build/html` (ensure `pip install ghp-import`)

## Code
- use [black](https://pypi.org/project/black/), [isort](https://github.com/timothycrosley/isort) for formatting
