for i in {0..7}
do
python similarity.py --device "$i" --batch_num "$i" /data/bin_data test.dict.json  &
done
