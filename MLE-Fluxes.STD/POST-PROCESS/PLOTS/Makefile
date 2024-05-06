# NEMO-MLE-Fluxes.STD
# -------------------
# Commands used to make the plots in ./RES
#
default: means snapshots

means: means_1d means_5d

means_1d:
	python3 ./plot_means.py -f 1d -y 1966 1967

means_5d:
	python3 ./plot_means.py -f 5d -y 1966 1967

snapshots:
	python3 ./plot_snpsht.py -f 1d -y 1967 -m 8 -d 2


