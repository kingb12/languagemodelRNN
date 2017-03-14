expt_dir=$1

python ../log_to_json.py $expt_dir'/'$expt_dir'_logs.json' $expt_dir'/'$expt_dir'.log'
python \
../create_report.py \
'../report_notebooks/'$expt_dir'.ipynb' \
'../report_templates/EncDecReportTemplate.ipynb' \
$expt_dir'/'$expt_dir'.json' \
$expt_dir'/'$expt_dir'_logs.json'
