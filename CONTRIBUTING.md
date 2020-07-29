# Contributing

## Online Book
We are using [jupyter book](https://jupyterbook.org/intro.html) for the website. The book is located in `jupyter/`.  Install `pip install -r jupyter/requirements.txt`.

**writing**
- the writing is either done in jupyter notebooks (`*.ipynb`) or markdown files (`*.md`).
- the entry point for the writing the book is `jupyter/_toc.yml` which gives the table of content, i.e. which files go where. From there you can easily find the important files (`*.ipynb` or `*.md`) to change, or add new ones your own.
- writing is done using an advanced Markdown [MyST](https://jupyterbook.org/content/myst.html) see this [cheatsheet](https://jupyterbook.org/reference/cheatsheet.html?highlight=table) to get started.

**Publish and view**
- `pip install -r requirements.txt` and  `pip install -r jupyter/requirements.txt`
- run `jupyter-book build jupyter/`
- view locally by opening `file://[path]/jupyter/_build/html/intro.html` in your browser (replace `[path]`)
- **ONLY TO PUT IT PUBLIC** upload to github pages `ghp-import -n -p -f jupyter/_build/html` (ensure `pip install ghp-import`)
- If you don't see the changes try removing the cache (force reload) of your browser. If it still does not work remove `jupyter/_build/html` and build again.

## Code
- use [black](https://pypi.org/project/black/), [isort](https://github.com/timothycrosley/isort) for formatting
