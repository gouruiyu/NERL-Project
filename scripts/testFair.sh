noise_max=10

for noise_max in `seq ${noise_max}`;
do
    python testFair.py --noise_regulator=0.0
done