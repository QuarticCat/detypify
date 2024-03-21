#!/bin/zsh

INPUT_FILE=data/standard.typ
OUTPUT_FILE=data/tex_to_typ.csv
INPUR_FILE_URL='https://raw.githubusercontent.com/mitex-rs/mitex/919336210ad4fa45b4823d302b6819625bdf29c4/packages/mitex/specs/latex/standard.typ'

setopt re_match_pcre

mkdir -p data

if [[ ! -r $INPUT_FILE ]] {
    echo 'Downloading standard.typ from mitex-rs/mitex...'
    curl $INPUR_FILE_URL > $INPUT_FILE
}

input=$(<$INPUT_FILE)
output=('latex,typst')

for title in Greeks 'Function symbols' Limits Symbols; {
    [[ $input =~ "// $title\n([\s\S]*?)\s+//" ]]
    section=$match[1]
    for line in ${(f)section}; {
        [[ $line =~ '"?(\w*)"?: (.*),' ]]
        tex_sym=$match[1]
        process=$match[2]
        case $process {
            (sym)         output+=("\\$tex_sym,$tex_sym") ;;
            (define-sym*) output+=("\\$tex_sym,${process[13,-3]//\\\\/\\}") ;;
            # TODO: handle of-sym, define-cmd and more
        }
    }
}

printf '%s\n' $output > $OUTPUT_FILE
