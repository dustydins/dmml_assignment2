#!/usr/bin/env bash

number=$1
shift
for i in `seq $number`; do
    echo RUN $i of $number
    $@
done
