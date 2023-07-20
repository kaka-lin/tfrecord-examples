#!/bin/bash
if [ -z "$1" ]
  then
    # navigate to ~/data
    echo "navigating to ./data/ ..."
    cd ./data/
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

echo "Now downloading Figaro1k.zip ..."

# wget http://projects.i-ctm.eu/sites/default/files/AltroMateriale/207_Michele%20Svanera/Figaro1k.zip
# The official link is not working for some reason, so temporarily use dropbox instead.
# wget https://www.dropbox.com/s/35momrh68zuhkei/Figaro1k.zip
wget http://projects.i-ctm.eu/sites/default/files/AltroMateriale/207_Michele%20Svanera/Figaro1k.zip

echo "Unzip Figaro1k.zip ..."

unzip Figaro1k.zip
mv Figaro1k figaro1k && mv figaro1k/Original figaro1k/images && mv figaro1k/GT figaro1k/masks
mv figaro1k/images/Training figaro1k/images/train
mv figaro1k/images/Testing figaro1k/images/test
mv figaro1k/masks/Training figaro1k/masks/train
mv figaro1k/masks/Testing figaro1k/masks/test

echo "Removing unnecessary files ..."

rm -f Figaro1k.zip
rm -f Figaro1k/masks/train/*'(1).pbm'
rm -f Figaro1k/.DS_Store
rm -rf __MACOSX

echo "Finished!"
