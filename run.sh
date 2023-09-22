#!/bin/sh

# Handle quotes strings in args
C=''
whitespace="[[:space:]]"
for i in "$@"
do
    if [[ $i =~ $whitespace ]]
    then
        i=\"$i\"
    fi 
    C="$C $i"
done
 
sh -c "python3 app/app.py $C"
