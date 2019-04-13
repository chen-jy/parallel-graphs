#include <cstdio>
#include <omp.h>
#include <string>
#include "getopt.h"

using namespace std;

int main(int argc, char *argv[]) {
	int tid, nthreads;
	string input_file;
	if (argc < 3) {
		printf("Usage: %s -n <nthreads> -f <filename>\n", argv[0]);
		return 1;
	}

	char option;
	while ((option = getopt(argc, argv, "n:f:")) != -1) {
		switch (option) {
		case 'n':
			nthreads = atoi(optarg);
			break;
		case 'f':
			input_file = string(optarg);
			break;
		default:
			printf("Usage: %s -n <nthreads> -f <filename>\n", argv[0]);
			return 1;
		}
	}

	return 0;
}
