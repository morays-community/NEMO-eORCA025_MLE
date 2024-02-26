#!/bin/bash
#
# ./autorun.sh <start_year> <end_year> <xios_freq>
#
# This script automate the submission of DMONTOOLS jobs to compute all yearly means from <start_year> to <end_year> with parameters in config_<xios_freq>
#

# Check arguments number and format
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <start_year> <end_year> <xios_freq>"
    exit 1
fi

# are they integer
if ! [[ $1 =~ ^[0-9]+$ ]] || ! [[ $2 =~ ^[0-9]+$ ]]; then
    echo "Year arguments must be integers."
    exit 1
fi

# are they in right order
if [ "$1" -ge "$2" ]; then
    echo "Start year must be lower than end year."
    exit 1
fi

# get config_moy
if ! cp config_moy_$3 config_moy; then
    echo "Error copying config_moy_$3. Exiting."
    exit 1
fi

# loop on years
for ((i = $1; i <= $2; i++)); do
    echo "Submitting year $i"
    ./RUN_calmoy $i
done
