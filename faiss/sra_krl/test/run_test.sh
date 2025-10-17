export OMP_NUM_THREADS=1
rm -rf ./coverage_report
rm coverage.info
make all -j
mkdir -p indices
./negative_distance.x
./dense_distance.x
./sparse_distance.x
./lut4b.x
./lut8b.x
./handle_IO.x
./reorder.x
lcov --capture --directory ./ --output-file coverage.info --exclude '*test*' --exclude '*12*' --rc lcov_branch_coverage=1
genhtml coverage.info --output-directory coverage_report --rc lcov_branch_coverage=1