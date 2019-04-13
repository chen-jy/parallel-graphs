#include <cstdio>
#include <omp.h>
#include <string>
#include <vector>
#include "getopt.h"

// Process input in an offline/online style
#define ONLINE 0
#define OFFLINE 1

using namespace std;

int main(int argc, char *argv[]) {
	int tid, nthreads, mode;
	string input_file;
	if (argc < 4) {
		printf("Usage: %s -n <nthreads> -f <filename> -m <mode>\n", argv[0]);
		exit(1);
	}

	char option;
	while ((option = getopt(argc, argv, "n:f:m:")) != -1) {
		switch (option) {
		case 'n':
			nthreads = atoi(optarg);
			break;
		case 'f':
			input_file = string(optarg);
			break;
		case 'm':
			mode = atoi(optarg);
			break;
		default:
			printf("Usage: %s -n <nthreads> -f <filename>\n", argv[0]);
			exit(1);
		}
	}

	omp_set_num_threads(nthreads);

	FILE *input = fopen(input_file.c_str(), "r");
	int N, M;
	fscanf(input, "%d %d", &N, &M);

	vector<vector<pair<int, double>>> graph(N);

	if (mode == ONLINE) {
		for (int i = 0; i < M; i++) {
			int u, v;
			double w;
			fscanf(input, "%d %d %lf", &u, &v, &w);
			graph[u].push_back({ v, w });
		}
	}
	else if (mode == OFFLINE) {
		#pragma omp parallel private(tid)
		{
			tid = omp_get_thread_num();
		}
	}
	else exit(2);

	return 0;
}
