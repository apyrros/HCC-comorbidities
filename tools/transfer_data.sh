#! /bin/bash

for d in $(ls $1)
do
    echo $d
    for sub_dir in $(ls $1/$d)
    do
        if $(test "$(ls -A $1/$d/$sub_dir)")
        then
            mv $1/$d/$sub_dir/*.png $1/$d.png
        fi
    done
    rm -r $1/$d
done
