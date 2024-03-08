#!/bin/sh

set -e
set -x

img_root="$1"
info_json="$2"
out_file="$3"
class_name="$4"
converted_file="$5"

python tools/ilsvrc_info.py --img-root ${img_root} --out-file ${info_json}
python tools/ilsvrc_folder.py --out-file ${out_file} --info-json ${info_json}
python tools/convert_ilsvrc_classes_name.py --ann ${out_file} --f ${class_name} --output ${converted_file}
