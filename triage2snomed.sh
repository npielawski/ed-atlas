#!/bin/bash
gunzip -c data/mimic/mimic-iv-ed/2.2/ed/triage.csv.gz \
| mlr --icsv --opprint --headerless-csv-output cut -f chiefcomplaint \
| awk '{print $0 "\n~"}' \
| sed 's/, */\n/g' \
| tr '[:upper:]' '[:lower:]' \
| sed 's/"//g' \
| sed "s/'//g" \
| sed "s/^\?//g" \
| sed 's/  */ /g' \
| sed 's/^ //g' \
| sed 's/abdominal/abd/g' \
| sed 's/^l //g' \
| sed 's/^r //g' \
| sed 's/^llq //g' \
| sed 's/^luq //g' \
| sed 's/^rlq //g' \
| sed 's/^ruq //g' \
| sed 's/^left //g' \
| sed 's/^right //g' \
| sed 's/^left sided //g' \
| sed 's/^right sided //g' \
| sed 's/^b //g' \
| sed 's/^bilat //g' \
| sed 's/^bilateral //g' \
| sed 's/^side //g' \
| sed 's/^mid //g' \
| sed 's/s\/p/status post/g' \
| sed 's/n\/v\/d/nausea vomiting diarrhea/g' \
| sed 's/n\/v/nausea vomiting/g' \
| sed 's/^mvc/motor vehicle collision/g' \
| sed 's/^ili/influenza like illness/g' \
| sed 's/^etoh/alcohol/g' \
| sed 's/^si/suicidal ideation/g' \
| sed 's/^brbpr/bright red blood per rectum/g' \
| sed 's/^vaginal/vag/g' \
| sed 's/^sob/shortness of breath/g' \
| sed 's/^cp/chest pain/g' \
| sed 's/^sdh/subdural hematoma/g' \
| sed 's/^ich/intracranial hemorrhage/g' \
| sed 's/^sah/subarachnoid hemorrhage/g' \
| sed 's/^cva/cerebrovascular accident/g' \
| sed 's/^sbo/small bowel obstruction/g' \
| sed 's/^ha/hematuria/g' \
| sed 's/\+ *//g' \
| sed 's/___ *//g' \
| sed 's/^ *\/ *//g' \
| sed 's/^ *\- *//g' \
| sed 's/abnml/abnormal/g' \
| sed 's/abnl/abnormal/g' \
| awk -v RS='~\n' -v ORS='\n' '{gsub(/\n/, ", "); print}' \
| sed 's/, *$//' \
> datasets/mimic_symptoms.txt
