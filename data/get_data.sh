#!/usr/bin/env bash

while read url; do
  echo "== $url =="
  curl -sL -O "$url"
done < list_of_urls.txt
