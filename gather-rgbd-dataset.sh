#!/bin/bash

dry_run=0
dataset_url=https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset
classes=(camera flashlight lightbulb pitcher stapler)
classes_files=(3 5 4 3 8)

if [ $dry_run == 1 ]
then
    cmd="echo"
else
    cmd=""
fi

# Create dataset directory
$cmd mkdir -p dataset/

cd dataset
classes_idx=0
for class in ${classes[@]}; do
    idx=1
    classes_files_idx=${classes_files[@]:$classes_idx:1} 
    while [ $idx -le $classes_files_idx ]
    do
	if [ ! -f ${class}_${idx} ]
	then
            $cmd curl -O ${dataset_url}/${class}_${idx}.tar
	fi
	$cmd tar -xf ${class}_${idx}.tar
	((idx++))
    done
    ((classes_idx++))
done

