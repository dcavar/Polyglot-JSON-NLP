#!/usr/bin/env bash

echo "downloading polyglot models"
echo "https://polyglot.readthedocs.io/en/stable/Download.html#langauge-task-support for all supported"
polyglot download LANG:en
polyglot download LANG:de
polyglot download LANG:es
polyglot download LANG:zh
polyglot download LANG:ar
polyglot download LANG:hi
polyglot download sgns2.en