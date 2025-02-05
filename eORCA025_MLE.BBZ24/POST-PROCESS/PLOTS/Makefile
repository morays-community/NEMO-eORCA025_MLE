# NEMO-MLE-FLuxes.CNN 
# -------------------
# Commands used to make the plots in ./RES
#
default: snapshots means

means: means_1d

snapshots:
	python3 ./plot_snpsht.py -f 1d -y 1966 -m 01 -d 07

means_1d:
	python3 ./plot_means.py -f 1d -y 1967 -m 3 9

diffmle:
	python3 ./diff_mle.py -cref eORCA025.L75-REF -f 1d -y 1967 -m 3
	python3 ./diff_mle.py -cref eORCA025.L75-REF -f 1d -y 1967 -m 9
