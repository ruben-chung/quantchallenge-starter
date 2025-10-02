#!/bin/bash

read -p "Enter time duration for interact session (e.g., 2:00:00): " time_input

interact -q gpu -g 1 -f ampere -m 20g -n 4 -t $time_input

