#!/usr/bin/perl -pi

# delete trailing spaces
# doesn't touch lines that are only spaces
s/(\S) +$/$1/;
