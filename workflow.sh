from terminal 1:

cd disclinations
pip freeze > requirements.txt
cd ..

from terminal 2:

docker run --platform linux/amd64 -v $(pwd):/workspace -w /workspace -it python:3.9 bash
pip download -r disclinations/requirements.txt -d disclinations_pkg_offline
pip download -r workspace/requirements.txt -d disclinations_pkg_offline
cd ..
rsync -av --exclude='.git' --exclude='.venv' --exclude='.pytest_cache' \
     --exclude='.virtual' --exclude='.ruff_cache' \
     disclinations/ disclinations_pkg_offline/disclinations/













from terminal 1:

tar -czvf disclinations_offline.tar.gz disclinations_pkg_offline
scp disclinations_offline.tar.gz leonbala@irene-fr.ccc.cea.fr:/ccc/work/cont003/gen14282/leonbala/     