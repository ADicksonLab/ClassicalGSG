#!/bin/bash --login


mol2path="database/Star_NonStar/"

mkdir "$mol2path/str"

search_dir="$mol2path/mol2"
for entry in "$search_dir"/*
do
		filename="${entry##*/}"
		basename="${filename%.*}"
		cgenff "$entry" >> "$mol2path/str/$basename.str"
		echo "$basname done"
done
