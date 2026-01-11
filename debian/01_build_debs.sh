#!/bin/bash
export DEBFULLNAME="Kenji Koide"
export DEBEMAIL="k.koide@aist.go.jp"

pushd .

cd ..

dch -v $1-1ppa1~jammy1 $2 -D jammy
git add debian/changelog
git commit -m "update debian/changelog for jammy"
gbp buildpackage -S -sa --git-upstream-tag='4.3a0' --git-submodules --git-force-create

dch -v $1-1ppa1~noble1 $2 -D noble
git add debian/changelog debian/files
git commit -m "update debian/changelog for noble"
gbp buildpackage -S -sa --git-upstream-tag='4.3a0' --git-submodules

popd
