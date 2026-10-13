"""Build install/source ZIPs without repository data or caches."""
from pathlib import Path
import re
import zipfile

root = Path(__file__).resolve().parents[1]
version = re.search(r'^version = "([^"]+)"', (root/'pyproject.toml').read_text(), re.MULTILINE).group(1)
files = [root/name for name in ('__init__.py','pyproject.toml','requirements.txt','LICENSE','README.md','SECURITY.md','.comfyignore')]
for directory in ('modules','web','assets'):
    files.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
source_files = files + [root/'.gitignore', root/'CONTRIBUTING.md']
for directory in ('tests','scripts'):
    source_files.extend(p for p in (root/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
(root/'dist').mkdir(exist_ok=True)
for suffix, paths in (('',files),('-source',source_files)):
    destination=root/'dist'/f'comfyui-cyberdelia-metadata-{version}{suffix}.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            z.write(p,Path(root.name)/p.relative_to(root))
    print(destination)
