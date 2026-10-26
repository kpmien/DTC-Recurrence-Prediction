"""Convert a '# %%' percent-format script into a Jupyter notebook."""
import sys, re, nbformat

src = open(sys.argv[1], encoding='utf-8').read()
cells = []
for block in re.split(r'^# %%', src, flags=re.M):
    if not block.strip():
        continue
    if block.startswith(' [markdown]'):
        body = block[len(' [markdown]'):].strip('\n')
        md = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(md.strip()))
    else:
        cells.append(nbformat.v4.new_code_cell(block.strip('\n')))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbformat.write(nb, sys.argv[2])
print(f'{len(cells)} cells -> {sys.argv[2]}')
